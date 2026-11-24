"""
Energy index per support phase (single support / double support / flight) for g1_decimation*_v2.

Contact is read at EVERY physics substep (200 Hz) so the phase split does not depend on the
policy rate. Nominal condition only (vx cmd, no push/payload/delay, noise 1x).

  CUDA_VISIBLE_DEVICES=1 python legged_gym/scripts/energy_gait_analysis.py --task=g1_decimation5_v2 --headless --out eval_results_v2_energy

Output: <out>/<task>.csv  (one row per env per repeat, only envs that survive the whole episode)
        <out>/<task>_meta.json
"""
import argparse
import json
import os
import sys

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--out", default="eval_results_v2_energy")
_p.add_argument("--episode_s", type=float, default=20.0)
_p.add_argument("--repeats", type=int, default=2)
_p.add_argument("--vx", type=float, default=0.6)
_own, _rest = _p.parse_known_args()
sys.argv = [sys.argv[0]] + _rest

import isaacgym  # noqa: E402
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

FORCE_THR = 1.0  # N, same threshold as the reward code
MIN_AIR_S = 0.05  # a touchdown counts as a step only after this time off the ground (rejects contact chatter)
CHORD_S = 0.8     # one gait period: distance = sum of pelvis displacements over this interval


class PhaseMeter:
    """Wraps env._compute_torques; accumulates power/torque by contact phase at every substep."""

    def __init__(self, env, groups):
        self.env = env
        self._orig = env._compute_torques
        env._compute_torques = self
        self.groups = groups  # dict name -> LongTensor of dof indices
        self.clear()
        self.reset_cycle()

    NB, BW = 60, 0.02   # cycle profile: 60 bins x 20 ms of time since left-foot touchdown

    def reset_cycle(self):
        dev = self.env.device
        self.cyc = torch.zeros(7, self.NB, device=dev)   # count, Ppos, Pneg, DS, Lcontact, Rcontact, tau2
        self.t_ltd = torch.full((self.env.num_envs,), -1.0, device=dev)

    def clear(self):
        n, dev = self.env.num_envs, self.env.device
        z = lambda *s: torch.zeros(*s, device=dev)
        self.n = z(n)                       # substeps
        self.cnt = z(n, 3)                  # [DS, SS, flight] substep counts
        self.p_pos = z(n, 3)                # sum of positive joint power per phase [W]
        self.p_neg = z(n, 3)                # sum of negative joint power (magnitude) per phase [W]
        self.tau2 = z(n, 3)                 # sum of sum_j tau^2 per phase
        self.grp_pos = z(n, len(self.groups))
        self.td = z(n, 2)                   # touchdowns per foot
        self.td_force = z(n)                # sum of Fz at touchdown substeps
        self.zhi = z(n) - 9.0                # pelvis height extrema
        self.zlo = z(n) + 9.0
        self.prev_c = getattr(self, "prev_c", None)
        self.air = getattr(self, "air", z(n, 2))

    def __call__(self, actions):
        env = self.env
        env.gym.refresh_net_contact_force_tensor(env.sim)
        fz = env.contact_forces[:, env.feet_indices, 2]
        c = fz > FORCE_THR                                     # [n,2]
        if self.prev_c is None:
            self.prev_c = c.clone()
        tau = self._orig(actions)
        p = tau * env.dof_vel                                  # [n,12]
        ncon = c.sum(1)                                        # 2 -> DS, 1 -> SS, 0 -> flight
        ph = torch.where(ncon == 2, 0, torch.where(ncon == 1, 1, 2))
        oh = torch.nn.functional.one_hot(ph, 3).float()        # [n,3]
        self.cnt += oh
        self.p_pos += oh * torch.clamp(p, min=0).sum(1, keepdim=True)
        self.p_neg += oh * torch.clamp(-p, min=0).sum(1, keepdim=True)
        self.tau2 += oh * (tau ** 2).sum(1, keepdim=True)
        for gi, idx in enumerate(self.groups.values()):
            self.grp_pos[:, gi] += torch.clamp(p[:, idx], min=0).sum(1)
        new = c & ~self.prev_c & (self.air >= MIN_AIR_S)
        self.air = torch.where(c, torch.zeros_like(self.air), self.air + env.sim_params.dt)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        h = env.rigid_body_states_view[:, 0, 2]
        self.zhi = torch.maximum(self.zhi, h)
        self.zlo = torch.minimum(self.zlo, h)
        # gait-cycle profile referenced to left-foot touchdown
        self.t_ltd = torch.where(new[:, 0], torch.zeros_like(self.t_ltd),
                                 torch.where(self.t_ltd >= 0, self.t_ltd + self.env.sim_params.dt, self.t_ltd))
        v = self.t_ltd >= 0
        b = torch.clamp((self.t_ltd / self.BW).long(), 0, self.NB - 1)
        vals = torch.stack([torch.ones_like(self.t_ltd), torch.clamp(p, min=0).sum(1), torch.clamp(-p, min=0).sum(1),
                            (ncon == 2).float(), c[:, 0].float(), c[:, 1].float(), (tau ** 2).sum(1)])
        for ri in range(7):
            self.cyc[ri].scatter_add_(0, b[v], vals[ri][v])
        self.td += new.float()
        self.td_force += (new.float() * fz).sum(1)
        self.prev_c = c.clone()
        self.n += 1
        return tau


@torch.no_grad()
def run_repeat(env, policy, meter, T):
    n, dev = env.num_envs, env.device
    n_steps = int(round(T / env.dt))
    obs, _ = env.reset()
    meter.prev_c = None
    meter.air = torch.zeros(n, 2, device=dev)
    meter.clear()
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    keys = ["cnt", "p_pos", "p_neg", "tau2", "grp_pos", "td", "td_force", "n"]
    tot = {k: torch.zeros_like(getattr(meter, k)) for k in keys}
    steps = torch.zeros(n, device=dev)
    dact = torch.zeros(n, device=dev)
    vx = torch.zeros(n, device=dev)
    zmax = torch.zeros(n, device=dev)
    zmin = torch.full((n,), 9.0, device=dev)
    prev_a = torch.zeros(n, env.num_actions, device=dev)
    x0 = env.root_states[:, 0].clone()
    p_prev = env.root_states[:, :2].clone()
    chord = torch.zeros(n, device=dev)
    chord_every = max(1, int(round(CHORD_S / env.dt)))
    for it in range(n_steps):
        a = policy(obs.detach())
        obs, _, _, dones, _ = env.step(a.detach())
        m = alive & ~dones.bool()
        mf = m.float()
        steps += mf
        for k in keys:
            v = getattr(meter, k)
            w = mf if v.dim() == 1 else mf.unsqueeze(1)
            tot[k] += v * w
        dact += (a - prev_a).abs().mean(1) * mf
        vx += env.base_lin_vel[:, 0] * mf
        zmax = torch.where(m, torch.maximum(zmax, meter.zhi), zmax)
        zmin = torch.where(m, torch.minimum(zmin, meter.zlo), zmin)
        if (it + 1) % chord_every == 0 or it == n_steps - 1:
            chord += (env.root_states[:, :2] - p_prev).norm(dim=1) * mf
            p_prev = env.root_states[:, :2].clone()
        prev_a = a.clone()
        meter.clear()
        alive &= ~dones.bool()
        if not alive.any():
            break
    dist = dict(chord=chord, world_x=env.root_states[:, 0] - x0)
    return tot, steps, dact, vx, dist, alive, zmax - zmin


def main():
    args = get_args()
    n_envs = args.num_envs or 1000
    args.num_envs = n_envs
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = n_envs
    env_cfg.env.episode_length_s = _own.episode_s + 5.0
    env_cfg.env.test = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1e5
    r = env_cfg.commands.ranges
    r.lin_vel_x = [_own.vx, _own.vx]
    r.lin_vel_y = [0.0, 0.0]
    r.ang_vel_yaw = [0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    names = list(env.dof_names)
    def idx(keys):
        return torch.tensor([i for i, nme in enumerate(names) if any(k in nme for k in keys)], device=env.device)
    groups = {"hip": idx(["hip"]), "knee": idx(["knee"]), "ankle": idx(["ankle"])}
    print("dof groups:", {k: v.tolist() for k, v in groups.items()}, names)

    props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])
    mass = float(sum(p.mass for p in props))
    meter = PhaseMeter(env, groups)
    run_repeat(env, policy, meter, 2.0)   # warm-up
    meter.reset_cycle()

    dt = env.sim_params.dt
    rows = []
    for rep in range(_own.repeats):
        tot, steps, dact, vx, dist, alive, hrange = run_repeat(env, policy, meter, _own.episode_s)
        ok = alive.cpu().numpy()
        t = (tot["n"] * dt).cpu().numpy()                    # seconds simulated
        cnt = tot["cnt"].cpu().numpy()
        pp, pn, t2 = tot["p_pos"].cpu().numpy(), tot["p_neg"].cpu().numpy(), tot["tau2"].cpu().numpy()
        gp = tot["grp_pos"].cpu().numpy()
        td = tot["td"].cpu().numpy()
        tdf = tot["td_force"].cpu().numpy()
        d = dist["chord"].cpu().numpy()     # distance along the path, independent of heading drift
        extra = {k: v.cpu().numpy() for k, v in dist.items()}
        for e in np.where(ok)[0]:
            T = t[e]
            nsub = tot["n"][e].item()
            row = dict(task=args.task, decimation=env_cfg.control.decimation, control_hz=1.0 / env.dt,
                       repeat=rep, env=int(e), dist_m=d[e], vx_mean=d[e] / T, mass_kg=mass,
                       dist_world_x_m=extra["world_x"][e])
            for i, nm in enumerate(["DS", "SS", "FL"]):
                row[f"frac_{nm}"] = cnt[e, i] / nsub
                row[f"E_pos_{nm}_J"] = pp[e, i] * dt             # positive mechanical work in phase
                row[f"E_neg_{nm}_J"] = pn[e, i] * dt             # negative work (magnitude) in phase
                row[f"tau2_{nm}"] = t2[e, i] * dt                # int sum tau^2 dt  (Joule-heating proxy)
                row[f"Ppos_{nm}_W"] = pp[e, i] / max(cnt[e, i], 1)
                row[f"Pneg_{nm}_W"] = pn[e, i] / max(cnt[e, i], 1)
            for gi, nm in enumerate(groups):
                row[f"E_pos_{nm}_J"] = gp[e, gi] * dt
            row["E_pos_J"] = pp[e].sum() * dt
            row["E_neg_J"] = pn[e].sum() * dt
            row["E_abs_J"] = row["E_pos_J"] + row["E_neg_J"]     # |tau*qd|, no regeneration
            row["E_joule_proxy"] = t2[e].sum() * dt
            row["steps_per_s"] = td[e].sum() / T
            row["step_len_m"] = d[e] / max(td[e].sum(), 1)
            row["touchdown_Fz_N"] = tdf[e] / max(td[e].sum(), 1)
            row["dact_per_step"] = dact[e].item() / steps[e].item()
            row["height_range_m"] = hrange[e].item()
            row["EPD_pos_J_per_m"] = row["E_pos_J"] / max(d[e], 0.1)     # energy per distance
            row["EPD_abs_J_per_m"] = row["E_abs_J"] / max(d[e], 0.1)
            row["CoT_pos"] = row["EPD_pos_J_per_m"] / (mass * 9.81)
            row["CoT_abs"] = row["EPD_abs_J_per_m"] / (mass * 9.81)
            rows.append(row)
        print(f"rep {rep}: survive={ok.mean():.3f}", flush=True)

    os.makedirs(_own.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(_own.out, f"{args.task}.csv"), index=False)
    with open(os.path.join(_own.out, f"{args.task}_meta.json"), "w") as f:
        json.dump(dict(task=args.task, decimation=env_cfg.control.decimation, control_hz=1.0 / env.dt,
                       mass_kg=mass, dof_names=names, force_thr_N=FORCE_THR, sim_dt=dt, vx_cmd=_own.vx,
                       episode_s=_own.episode_s, repeats=_own.repeats), f, indent=2)
    np.save(os.path.join(_own.out, f"{args.task}_cycle.npy"), meter.cyc.cpu().numpy())
    if len(df):
        print(df[["frac_DS", "frac_SS", "frac_FL", "EPD_pos_J_per_m", "CoT_pos", "steps_per_s"]].mean())


if __name__ == "__main__":
    main()
