"""Foot-trajectory figure and swing statistics from foot_trajectory.py output.

A swing is one interval between lift-off and touchdown of a foot; intervals shorter than MIN_AIR_S are contact
chatter and are counted separately. Displacement is measured along the pelvis heading at mid-swing.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "eval_results_v2_energy"
D, OUT = f"{ROOT}/foot", f"{ROOT}/figs"
DECS = [40, 36, 32, 28, 24, 20, 16, 12, 8, 5, 4, 3, 2, 1]
SHOW = [40, 32, 24, 8, 4]
MIN_AIR_S = 0.05


def load(dec):
    z = np.load(f"{D}/g1_decimation{dec}_v2_foot.npz")
    return {k: z[k] for k in z.files}


def swings(d):
    """-> array [clearance m, duration s, displacement m] per swing, and the duration of every airborne interval."""
    dt, c = float(d["dt"]), d["contact"] > 0.5
    foot, yaw = d["foot"], np.unwrap(d["yaw"], axis=0)
    res, air = [], []
    for e in np.where(d["alive"])[0]:
        for f in range(2):
            ce = c[:, e, f]
            z0 = np.median(foot[ce, e, f, 2])
            edge = np.diff(ce.astype(int))
            off, on = np.where(edge == -1)[0] + 1, np.where(edge == 1)[0] + 1
            for s in off:
                nxt = on[on > s]
                if len(nxt) == 0:
                    break
                t = nxt[0]
                air.append((t - s) * dt)
                if air[-1] < MIN_AIR_S:
                    continue
                m = (s + t) // 2
                dd = foot[t, e, f, :2] - foot[s, e, f, :2]
                res.append((foot[s:t, e, f, 2].max() - z0, (t - s) * dt,
                            dd[0] * np.cos(yaw[m, e]) + dd[1] * np.sin(yaw[m, e]), f))
    return np.array(res), np.array(air)


data, rows = {}, []
for dec in DECS:
    d = load(dec)
    s, air = swings(d)
    data[dec] = (d, s)
    dt, al = float(d["dt"]), d["alive"]
    T = len(d["root"]) * dt
    k = int(round(0.8 / dt))
    L, R = s[s[:, 3] == 0], s[s[:, 3] == 1]
    xy = d["root"][::k, :, :2][:, al]
    yaw = np.unwrap(d["yaw"], axis=0)[:, al]
    q = lambda a, p: float(np.percentile(a, p))
    rows.append(dict(control_hz=float(d["control_hz"]), n_robots=int(al.sum()), n_swings=len(s),
                     path_speed_mps=float(np.linalg.norm(np.diff(xy, axis=0), axis=2).sum(0).mean() / ((len(xy) - 1) * k * dt)),
                     world_x_speed_mps=float((d["root"][-1, al, 0] - d["root"][0, al, 0]).mean() / T),
                     abs_yaw_drift_deg=float(np.degrees(np.abs(yaw[-1] - yaw[0])).mean()),
                     swings_per_s=len(s) / T / al.sum(), chatter_frac=float((air < MIN_AIR_S).mean()),
                     clear_p10_cm=q(s[:, 0], 10) * 100, clear_p50_cm=q(s[:, 0], 50) * 100, clear_p90_cm=q(s[:, 0], 90) * 100,
                     clear_below_2cm=float((s[:, 0] < 0.02).mean()), swing_p50_ms=q(s[:, 1], 50) * 1000,
                     disp_p10_cm=q(s[:, 2], 10) * 100, disp_p50_cm=q(s[:, 2], 50) * 100, disp_p90_cm=q(s[:, 2], 90) * 100,
                     disp_below_10cm=float((s[:, 2] < 0.10).mean()),
                     clear_left_cm=q(L[:, 0], 50) * 100, clear_right_cm=q(R[:, 0], 50) * 100,
                     swing_left_ms=q(L[:, 1], 50) * 1000, swing_right_ms=q(R[:, 1], 50) * 1000))
st = pd.DataFrame(rows)
st.to_csv(f"{ROOT}/foot_stats.csv", index=False)
print(st.round(2).to_string())

cols = ["#b2182b", "#ef8a62", "#e6ab02", "#1b7837", "#2166ac"]
fig, ax = plt.subplots(2, 2, figsize=(10, 6.6))
off = 12
for i, dec in enumerate(SHOW):
    d = data[dec][0]
    e = np.where(d["alive"])[0][0]
    t = np.arange(len(d["foot"])) * float(d["dt"])
    m = (t >= 10) & (t < 13)
    for f, ls in ((0, "-"), (1, "--")):
        z = d["foot"][:, e, f, 2]
        z0 = np.median(z[d["contact"][:, e, f] > 0.5])
        ax[0, 0].plot(t[m] - 10, (z[m] - z0) * 100 + off * (len(SHOW) - 1 - i), ls, color=cols[i], lw=1)
    ax[0, 0].text(3.16, off * (len(SHOW) - 1 - i), f"{d['control_hz']:.1f} Hz", color=cols[i], fontsize=8, va="bottom", clip_on=False)
    xy = d["root"][:, e, :2] - d["root"][0, e, :2]
    ax[0, 1].plot(xy[:, 0], xy[:, 1], color=cols[i], lw=1.4, label=f"{d['control_hz']:.1f} Hz")
d36 = data[36][0]
e = np.where(d36["alive"])[0][0]
xy = d36["root"][:, e, :2] - d36["root"][0, e, :2]
ax[0, 1].plot(xy[:, 0], xy[:, 1], color="0.2", lw=1.4, ls=":", label="5.6 Hz")
ax[0, 0].set_xlabel("Time (s)"); ax[0, 0].set_ylabel(f"Foot height (cm, offset {off} cm)")
ax[0, 0].set_title("(a) Foot height, left solid, right dashed", fontsize=10)
ax[0, 1].set_xlabel("x (m)"); ax[0, 1].set_ylabel("y (m)"); ax[0, 1].set_aspect("equal", adjustable="datalim")
ax[0, 1].set_title("(b) Pelvis path from above, 20 s", fontsize=10); ax[0, 1].legend(fontsize=7, frameon=False)
order = DECS
lab = [f"{data[x][0]['control_hz']:.1f}".replace(".0", "") for x in order]
for a, j, yl, ti in ((ax[1, 0], 0, "Swing apex clearance (cm)", "(c) Clearance per swing"),
                     (ax[1, 1], 2, "Displacement per swing (cm)", "(d) Displacement per swing")):
    a.boxplot([data[x][1][:, j] * 100 for x in order], labels=lab, showfliers=False, whis=(10, 90), medianprops=dict(color="#b2182b"))
    a.set_xlabel("Policy rate (Hz)"); a.set_ylabel(yl); a.set_title(ti, fontsize=10); a.tick_params(axis="x", labelsize=7)
for a in ax.ravel():
    a.grid(alpha=0.25)
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/foot_trajectory.png", dpi=200)
