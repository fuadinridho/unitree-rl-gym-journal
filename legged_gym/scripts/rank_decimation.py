"""Peringkat frekuensi kontrol dengan error bar antar-seed (hasil eval_decimation.py v2).
Skor robustness = rata-rata success di semua level semua sweep (push, payload, friction, delay_ms, obs_noise).
Usage: python legged_gym/scripts/rank_decimation.py [--dir eval_results_v2]"""
import argparse, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="eval_results_v2")
a = ap.parse_args()
d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(a.dir, "g1_decimation*_v2*.csv"))])
d["seed"] = d.task.str.extract(r"_s(\d+)$")[0].fillna(0).astype(int)
per = d[d.axis != "nominal"].groupby(["control_hz", "seed", "axis", "level"]).success.mean().reset_index()
rob = per.groupby(["control_hz", "seed"]).success.mean().rename("robust").reset_index()
nom = d[d.axis == "nominal"].groupby(["control_hz", "seed"])[["cot", "vel_rmse", "tilt_mean_deg"]].mean().reset_index()
m = rob.merge(nom)
g = m.groupby("control_hz")
agg = g.agg(n_seeds=("seed", "size"), robust=("robust", "mean"), robust_sd=("robust", "std"), cot=("cot", "mean"), cot_sd=("cot", "std"),
            vel_rmse=("vel_rmse", "mean"), vel_sd=("vel_rmse", "std"), tilt=("tilt_mean_deg", "mean")).sort_values("robust", ascending=False)
agg.round(4).to_csv(os.path.join(a.dir, "ranking.csv"))
print(agg.round(3).to_string())

fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
s = agg.sort_index()
for ax, (k, sd, t) in zip(axs, [("robust", "robust_sd", "Skor robustness (rata-rata success)"), ("cot", "cot_sd", "Cost of transport (nominal)"),
                                 ("vel_rmse", "vel_sd", "RMSE kecepatan xy (m/s)")]):
    ax.errorbar(s.index, s[k], yerr=s[sd].fillna(0), fmt="-o", color="#2a6fb0", ecolor="#6b7785", capsize=3, lw=2, ms=5)
    for hz, r in s.iterrows():
        if r.n_seeds == 1:
            ax.plot(hz, r[k], "o", mfc="white", color="#2a6fb0", ms=7)
    ax.set_xscale("log"); ax.set_xticks(s.index); ax.set_xticklabels([f"{h:g}" for h in s.index], rotation=60, fontsize=7)
    ax.minorticks_off(); ax.set_xlabel("frekuensi kontrol (Hz)"); ax.set_title(t, fontsize=10)
    ax.grid(alpha=0.3); [ax.spines[x].set_visible(False) for x in ("top", "right")]
fig.text(0.99, 0.01, "error bar = sd antar-seed (3 seed); lingkaran putih = 1 seed", ha="right", fontsize=7, color="#6b7785")
fig.tight_layout()
fig.savefig(os.path.join(a.dir, "plots", "ranking_seeds.png"), dpi=140)
print("saved", os.path.join(a.dir, "plots", "ranking_seeds.png"))
