"""
Statistics and figures for the energy / support-phase analysis.
Input : eval_results_v2_energy/<task>.csv  (energy_gait_analysis.py) + eval_results_v2/<task>.csv (nominal rows)
Output: eval_results_v2_energy/{freq_summary.csv, env_regression.csv, stats.json}
        eval_results_v2_energy/figs/*.png
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = sys.argv[1] if len(sys.argv) > 1 else "eval_results_v2_energy"
V2 = "eval_results_v2"
FIG = os.path.join(ROOT, "figs")
os.makedirs(FIG, exist_ok=True)
REF_HZ = 50.0   # common RL default; index = 1.00 there

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{ROOT}/g1_*.csv")) if "summary" not in f], ignore_index=True)
df["seed"] = df.task.map(lambda t: int(re.search(r"_s(\d+)$", t).group(1)) if re.search(r"_s(\d+)$", t) else 0)
df["P_pos_W"] = df.E_pos_J / (df.dist_m / df.vx_mean)
df["share_DS_E"] = df.E_pos_DS_J / df.E_pos_J
df["share_SS_E"] = df.E_pos_SS_J / df.E_pos_J
df["regen_frac"] = df.E_neg_J / df.E_abs_J
df["Ppos_ratio_DS_SS"] = df.Ppos_DS_W / df.Ppos_SS_W

# ---- per-task means, then per-frequency mean +- sd over seeds ----
num = [c for c in df.columns if df[c].dtype != object and c not in ("env", "repeat", "seed", "decimation", "control_hz")]
per_task = df.groupby(["task", "decimation", "control_hz", "seed"])[num].mean().reset_index()
g = per_task.groupby(["decimation", "control_hz"])
mean = g[num].mean()
sd = g[num].std()
nseed = g.size().rename("n_seeds")
summ = mean.join(sd, rsuffix="_sd").join(nseed).reset_index().sort_values("control_hz")
ref = summ.loc[(summ.control_hz - REF_HZ).abs() < 1, "CoT_pos"].iloc[0]
summ["REI_pos"] = summ.CoT_pos / ref                 # relative energy index (positive work), 1 at 50 Hz
refa = summ.loc[(summ.control_hz - REF_HZ).abs() < 1, "CoT_abs"].iloc[0]
summ["REI_abs"] = summ.CoT_abs / refa
summ.to_csv(f"{ROOT}/freq_summary.csv", index=False)

out = {}
WALK_MIN_HZ = 8.0   # below this: one seed per rate and large heading drift; kept out of the main statistics (see sensitivity_all_rates)
walk = per_task[per_task.control_hz >= WALK_MIN_HZ]

# ---- frequency trend of the energy index (Spearman on task-level data) ----
for col in ["CoT_pos", "CoT_abs", "EPD_pos_J_per_m", "frac_DS", "frac_SS", "frac_FL", "steps_per_s",
            "step_len_m", "touchdown_Fz_N", "regen_frac", "dact_per_step", "Ppos_SS_W"]:
    rho, p = stats.spearmanr(walk.control_hz, walk[col])
    out[f"spearman_hz_vs_{col}"] = dict(rho=float(rho), p=float(p))

# ---- task-level link between double-support share and energy, and the U-shaped optimum ----
for col in ["CoT_pos", "CoT_abs"]:
    rho, p = stats.spearmanr(walk.frac_DS, walk[col])
    out[f"spearman_DS_vs_{col}"] = dict(rho=float(rho), p=float(p), n=int(len(walk)))
rho, p = stats.spearmanr(walk.Ppos_DS_W / walk.Ppos_SS_W, walk.control_hz)
out["spearman_powerratio_DS_SS_vs_hz"] = dict(rho=float(rho), p=float(p))
lx = np.log10(walk.control_hz.values); yy = walk.CoT_pos.values
rng = np.random.default_rng(0); opt = []
for _ in range(2000):
    i = rng.integers(0, len(lx), len(lx))
    if len(np.unique(lx[i])) < 4:
        continue
    c = np.polyfit(lx[i], yy[i], 2)
    if c[0] > 0:
        opt.append(10 ** (-c[1] / (2 * c[0])))
c = np.polyfit(lx, yy, 2)
out["quad_fit_optimum_hz"] = dict(point=float(10 ** (-c[1] / (2 * c[0]))), ci95=[float(np.percentile(opt, 2.5)), float(np.percentile(opt, 97.5))],
                                  n_boot=len(opt), coef=[float(x) for x in c])
# sensitivity: same statistics with the rates below WALK_MIN_HZ included
c_all = np.polyfit(np.log10(per_task.control_hz), per_task.CoT_pos, 2)
r_all = stats.spearmanr(per_task.frac_DS, per_task.CoT_pos)
out["sensitivity_all_rates"] = dict(n=int(len(per_task)), quad_optimum_hz=float(10 ** (-c_all[1] / (2 * c_all[0]))),
                                    spearman_DS_vs_CoT=float(r_all[0]), p=float(r_all[1]))
# pairwise tests on seed-level CoT among the 3-seed frequencies
three = per_task.groupby("control_hz").filter(lambda d: len(d) == 3)
base = three[(three.control_hz - 25).abs() < 1].CoT_pos.values
for h in sorted(three.control_hz.unique()):
    v = three[three.control_hz == h].CoT_pos.values
    t = stats.ttest_ind(v, base, equal_var=False)
    out[f"welch_CoT_{h:.1f}Hz_vs_25Hz"] = dict(mean=float(v.mean()), diff=float(v.mean() - base.mean()), p=float(t.pvalue))

# ---- within-task regression: what makes ONE environment cheaper than another at the same policy ----
# step_len_m is left out: vx = step rate x step length, so the three are collinear
preds = ["frac_DS", "steps_per_s", "touchdown_Fz_N", "dact_per_step", "height_range_m", "vx_mean"]

rows = []
for t, d in df[df.control_hz >= WALK_MIN_HZ].groupby("task"):
    d = d.dropna(subset=preds + ["CoT_pos"])
    if len(d) < 100:
        continue
    y = (np.log(d.CoT_pos) - np.log(d.CoT_pos).mean()) / np.log(d.CoT_pos).std()
    X = d[preds].copy()
    X = (X - X.mean()) / X.std().replace(0, np.nan)
    X = X.fillna(0)
    beta, *_ = np.linalg.lstsq(np.c_[X.values, np.ones(len(X))], y.values, rcond=None)
    pred = np.c_[X.values, np.ones(len(X))] @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    row = dict(task=t, control_hz=d.control_hz.iloc[0], n=len(d), r2=float(r2))
    row.update({f"beta_{p}": float(b) for p, b in zip(preds, beta[:-1])})
    for p in ["frac_DS", "steps_per_s", "step_len_m", "touchdown_Fz_N", "dact_per_step", "height_range_m"]:
        rho, _ = stats.spearmanr(d[p], d.CoT_pos)
        row[f"rho_{p}"] = float(rho)
    rows.append(row)
reg = pd.DataFrame(rows).sort_values("control_hz")
reg.to_csv(f"{ROOT}/env_regression.csv", index=False)

# consistency across the 30 policies: sign of beta and a sign test
for p in preds:
    b = reg[f"beta_{p}"]
    pos = int((b > 0).sum())
    out[f"beta_{p}"] = dict(mean=float(b.mean()), sd=float(b.std()), n_pos=pos, n=int(len(b)),
                            sign_test_p=float(stats.binomtest(pos, len(b), 0.5).pvalue))

# ---- DS-rich vs DS-poor environments of the same policy (paired by task) ----
diffs = []
for t, d in df[df.control_hz >= WALK_MIN_HZ].groupby("task"):
    lo, hi = d.frac_DS.quantile([0.25, 0.75])
    a, b = d[d.frac_DS <= lo], d[d.frac_DS >= hi]
    if len(a) > 20 and len(b) > 20:
        diffs.append(dict(task=t, hz=d.control_hz.iloc[0], dCoT=b.CoT_pos.mean() / a.CoT_pos.mean() - 1,
                          dSteps=b.steps_per_s.mean() / a.steps_per_s.mean() - 1))
dd = pd.DataFrame(diffs)
if len(dd):
    w = stats.wilcoxon(dd.dCoT)
    out["DS_top_vs_bottom_quartile_dCoT"] = dict(mean=float(dd.dCoT.mean()), sd=float(dd.dCoT.std()),
                                                  wilcoxon_p=float(w.pvalue), n=int(len(dd)))

# ---- energy decomposition by phase at representative frequencies ----
for hz in summ.control_hz:
    r = summ[summ.control_hz == hz].iloc[0]
    out[f"phase_{hz:.1f}Hz"] = dict(frac_DS=float(r.frac_DS), frac_SS=float(r.frac_SS), frac_FL=float(r.frac_FL),
                                    share_DS_E=float(r.share_DS_E), share_SS_E=float(r.share_SS_E),
                                    Ppos_DS=float(r.Ppos_DS_W), Ppos_SS=float(r.Ppos_SS_W),
                                    CoT_pos=float(r.CoT_pos), REI=float(r.REI_pos))
json.dump(out, open(f"{ROOT}/stats.json", "w"), indent=1)

# ================= figures =================
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C = dict(DS="#d95f02", SS="#1b9e77", FL="#7570b3", ink="#333333", acc="#1f78b4", g="#999999")
hz = summ.control_hz.values
xt = dict(xscale="log")

def fmt(ax):
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "200"])
    ax.set_xlabel("Policy frequency (Hz)")

# F2: support-phase fractions and power per phase
fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
for k, nm in [("DS", "double support"), ("SS", "single support"), ("FL", "flight")]:
    ax[0].errorbar(hz, summ[f"frac_{k}"] * 100, yerr=summ[f"frac_{k}_sd"].fillna(0) * 100, marker="o", ms=4, color=C[k], label=nm, capsize=2)
fmt(ax[0]); ax[0].set_ylabel("Time share (%)"); ax[0].legend(frameon=False)
for k, nm in [("DS", "double support"), ("SS", "single support")]:
    ax[1].errorbar(hz, summ[f"Ppos_{k}_W"], yerr=summ[f"Ppos_{k}_W_sd"].fillna(0), marker="o", ms=4, color=C[k], label=nm, capsize=2)
fmt(ax[1]); ax[1].set_ylabel("Mean positive joint power in phase (W)"); ax[1].legend(frameon=False)
ax[2].bar(range(len(hz)), summ.share_DS_E * 100, color=C["DS"], label="double support")
ax[2].bar(range(len(hz)), summ.share_SS_E * 100, bottom=summ.share_DS_E * 100, color=C["SS"], label="single support")
ax[2].set_xticks(range(len(hz))); ax[2].set_xticklabels([f"{h:.0f}" for h in hz], rotation=60)
ax[2].set_xlabel("Policy frequency (Hz)"); ax[2].set_ylabel("Share of positive work (%)")
fig.tight_layout(); fig.savefig(f"{FIG}/phase_vs_hz.png"); plt.close(fig)

# F3: within-policy standardized regression coefficients
labs = {"frac_DS": "double-support share", "steps_per_s": "step rate", "step_len_m": "step length",
        "touchdown_Fz_N": "touchdown force", "dact_per_step": "action change", "height_range_m": "base height range",
        "vx_mean": "forward speed"}
fig, ax = plt.subplots(figsize=(5.2, 3))
ms = [reg[f"beta_{p}"].mean() for p in preds]
ss = [reg[f"beta_{p}"].std() for p in preds]
order = np.argsort(ms)
ax.barh(range(len(preds)), np.array(ms)[order], xerr=np.array(ss)[order], color=[C["acc"] if m > 0 else C["SS"] for m in np.array(ms)[order]], capsize=2)
ax.set_yticks(range(len(preds))); ax.set_yticklabels([labs[preds[i]] for i in order])
ax.axvline(0, color=C["ink"], lw=0.8)
ax.set_xlabel("Std. coefficient on log CoT (mean $\\pm$ sd over policies)")
fig.tight_layout(); fig.savefig(f"{FIG}/env_regression.png"); plt.close(fig)

# F6: gait-cycle profile
labsel = [(12.5, "12.5 Hz"), (25.0, "25 Hz"), (40.0, "40 Hz"), (100.0, "100 Hz")]
fig, ax = plt.subplots(2, 4, figsize=(10, 4.2), sharex=True, sharey="row")
prof = {}
for j, (h, lab) in enumerate(labsel):
    acc = None
    for t in per_task[(per_task.control_hz - h).abs() < 0.5].task:
        a = np.load(f"{ROOT}/{t}_cycle.npy")
        acc = a if acc is None else acc + a
    n = np.maximum(acc[0], 1)
    P = acc[1] / n
    stride = 2.0 / summ[(summ.control_hz - h).abs() < 0.5].steps_per_s.iloc[0]
    x = (np.arange(len(n)) + 0.5) * 0.02 / stride
    k = x <= 1.0
    ax[0, j].fill_between(x[k], 0, 1, where=(acc[3] / n)[k] > 0.5, color=C["DS"], alpha=0.18, step="mid", transform=ax[0, j].get_xaxis_transform())
    ax[0, j].plot(x[k], P[k], color=C["acc"], lw=1.4, label="positive")
    ax[0, j].plot(x[k], (acc[2] / n)[k], color=C["g"], lw=1.2, label="negative")
    ax[0, j].set_title(lab)
    ax[1, j].plot(x[k], (acc[4] / n)[k], color=C["SS"], lw=1.4, label="left foot")
    ax[1, j].plot(x[k], (acc[5] / n)[k], color=C["FL"], lw=1.4, label="right foot")
    ax[1, j].plot(x[k], (acc[3] / n)[k], color=C["DS"], lw=1.4, ls="--", label="both feet")
    ax[1, j].set_xlabel("Stride cycle (from left touchdown)")
    prof[lab] = dict(stride_s=float(stride), peak_W=float(P[k].max()), min_W=float(P[k].min()))
ax[0, 0].set_ylabel("Joint power (W)"); ax[1, 0].set_ylabel("Contact probability")
ax[0, 0].legend(frameon=False, fontsize=7); ax[1, 0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/cycle_profile.png"); plt.close(fig)
out["cycle_profile"] = prof
json.dump(out, open(f"{ROOT}/stats.json", "w"), indent=1)

# F4: gait descriptors
fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
for a, col, lab in zip(ax, ["steps_per_s", "step_len_m", "touchdown_Fz_N"], ["Steps per second", "Step length (m)", "Touchdown force (N)"]):
    a.errorbar(hz, summ[col], yerr=summ[col + "_sd"].fillna(0), marker="o", ms=4, color=C["acc"], capsize=2)
    fmt(a); a.set_ylabel(lab)
fig.tight_layout(); fig.savefig(f"{FIG}/gait_vs_hz.png"); plt.close(fig)

# F5: joint groups
fig, ax = plt.subplots(figsize=(5.2, 3))
tot = summ[["E_pos_hip_J", "E_pos_knee_J", "E_pos_ankle_J"]].sum(1)
bot = np.zeros(len(summ))
for k, c in [("hip", "#1f78b4"), ("knee", "#e6ab02"), ("ankle", "#66a61e")]:
    v = summ[f"E_pos_{k}_J"] / tot * 100
    ax.bar(range(len(hz)), v, bottom=bot, color=c, label=k); bot += v.values
ax.set_xticks(range(len(hz))); ax.set_xticklabels([f"{h:.0f}" for h in hz], rotation=60)
ax.set_xlabel("Policy frequency (Hz)"); ax.set_ylabel("Share of positive work (%)"); ax.legend(frameon=False, ncol=3)
fig.tight_layout(); fig.savefig(f"{FIG}/joint_share.png"); plt.close(fig)

print(summ[["control_hz", "n_seeds", "frac_DS", "frac_SS", "frac_FL", "Ppos_DS_W", "Ppos_SS_W", "share_DS_E",
            "CoT_pos", "CoT_abs", "REI_pos", "steps_per_s", "step_len_m", "touchdown_Fz_N", "regen_frac"]].round(3).to_string())
print(json.dumps({k: v for k, v in out.items() if not k.startswith("phase_")}, indent=1))
print(reg.round(2).to_string())

# ---- speed-adjusted CoT: CoT falls ~1/v, so policies that walk faster look cheaper; adjust to v_ref = 0.5 m/s ----
dw = df[(df.control_hz >= WALK_MIN_HZ) & (df.vx_mean > 0.1)].copy()
dw["ly"] = np.log(dw.CoT_pos); dw["lx"] = np.log(dw.vx_mean)
dw["ly"] -= dw.groupby("task").ly.transform("mean"); dw["lx"] -= dw.groupby("task").lx.transform("mean")
slope = float(np.polyfit(dw.lx, dw.ly, 1)[0])
pt = dw.groupby(["task", "control_hz"]).agg(c=("CoT_pos", "mean"), v=("vx_mean", "mean")).reset_index()
pt["adj"] = pt.c * (pt.v / 0.5) ** (-slope)
sa = pt.groupby("control_hz").agg(CoT_adj=("adj", "mean"), CoT_adj_sd=("adj", "std"), n=("adj", "size")).reset_index()
sa.to_csv(f"{ROOT}/speed_adjusted_cot.csv", index=False)
base = pt[(pt.control_hz - 25).abs() < 1].adj.values
sp = {"slope_dlogCoT_dlogv": slope}
for h in sorted(pt.control_hz.unique()):
    v = pt[pt.control_hz == h].adj.values
    if len(v) == 3:
        sp[f"{h:.1f}"] = dict(adj=float(v.mean()), diff_vs_25=float(v.mean() - base.mean()),
                              p=float(stats.ttest_ind(v, base, equal_var=False).pvalue))
out["speed_adjusted"] = sp
json.dump(out, open(f"{ROOT}/stats.json", "w"), indent=1)
# F1: energy index vs frequency (dashed line: lower limit of the main statistics)
sw = summ
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for col, lab, c in [("CoT_pos", "positive work", C["acc"]), ("CoT_abs", "absolute work", C["DS"])]:
    ax[0].errorbar(sw.control_hz, sw[col], yerr=sw[col + "_sd"].fillna(0), marker="o", ms=4, lw=1.4, color=c, label=lab, capsize=2)
ax[0].axvline(WALK_MIN_HZ, color="0.6", ls="--", lw=0.8)
fmt(ax[0]); ax[0].set_ylabel("Cost of transport (-)"); ax[0].legend(frameon=False)
ax[1].errorbar(sa.control_hz, sa.CoT_adj, yerr=sa.CoT_adj_sd.fillna(0), marker="o", ms=4, color=C["acc"], capsize=2)
fmt(ax[1]); ax[1].set_ylabel("Speed-adjusted CoT (-), $v_{ref}$=0.5 m/s")
fig.tight_layout(); fig.savefig(f"{FIG}/energy_index_vs_hz.png"); plt.close(fig)

print(sa.round(3).to_string()); print(sp)
