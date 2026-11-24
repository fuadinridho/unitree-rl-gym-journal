"""
Evaluasi satu policy g1_decimation* di Isaac Gym:
  1. protokol nominal (N env x R repeat, kondisi sama)
  2. sweep robustness (push, payload, friction, actuator delay, obs noise) satu faktor per sweep
  3. benchmark waktu komputasi inference policy

Contoh:
  CUDA_VISIBLE_DEVICES=1 python legged_gym/scripts/eval_decimation.py --task=g1_decimation20 --headless
  ... --num_envs 200 --repeats 1 --episode_s 10 --axes push   # smoke test cepat

Hasil: eval_results/<task>.csv (satu baris per axis/level/repeat) + <task>_meta.json

Catatan: env ini TIDAK punya objek/tray. "success" = robot tidak jatuh/terminasi
sampai akhir episode; kualitas platform (tray-proxy) diukur lewat kemiringan dan
getaran badan (tilt, tilt-rate, bounce).
"""
import argparse
import csv
import json
import os
import sys
import time

# argumen khusus dipisah dulu, sisanya diteruskan ke get_args() milik legged_gym
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--out", default="eval_results")
_p.add_argument("--episode_s", type=float, default=20.0, help="durasi evaluasi per episode [s]")
_p.add_argument("--repeats", type=int, default=3, help="berapa kali set N env dijalankan ulang")
_p.add_argument("--vx", type=float, default=0.6, help="perintah kecepatan maju [m/s]")
_p.add_argument("--axes", default="nominal,push,payload,delay_ms,obs_noise,friction",
                help="daftar sweep dipisah koma; friction harus TERAKHIR (level 0.1 mengontaminasi state PhysX)")
_p.add_argument("--bench_iters", type=int, default=2000)
_own, _rest = _p.parse_known_args()
sys.argv = [sys.argv[0]] + _rest

import isaacgym  # noqa: E402  (harus sebelum torch)
from isaacgym import gymapi  # noqa: E402
from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry, export_policy_as_jit  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Kondisi nominal = setting training (noise 1x) tanpa push/payload/delay, gesekan 1.0
NOMINAL = dict(push=0.0, payload=0.0, friction=1.0, delay_ms=0.0, obs_noise=1.0)
SWEEPS = {
    "push":      [0.5, 1.0, 1.5, 2.0, 3.0],        # kecepatan dorong maks xy [m/s], tiap 5 s
    "payload":   [2.0, 4.0, 6.0, 8.0, 10.0],       # massa tambahan di badan [kg]
    "friction":  [0.5, 0.35, 0.25, 0.15, 0.1],      # koefisien gesek efektif kontak kaki-lantai
    "delay_ms":  [5.0, 10.0, 20.0, 40.0, 80.0],    # latensi aktuator [ms]
    "obs_noise": [0.0, 2.0, 3.0, 4.0, 6.0],        # pengali noise observasi (1.0 = training)
}


class Instrument:
    """Membungkus env._compute_torques: delay aktuator + akumulasi energi per substep."""

    def __init__(self, env):
        self.env = env
        self._orig = env._compute_torques
        env._compute_torques = self
        self.k = 0
        self.clear()

    def set_delay(self, k_sub):
        self.k = int(k_sub)
        self.hist = None

    def clear(self):
        n = self.env.num_envs
        dev = self.env.device
        self.p_pos = torch.zeros(n, device=dev)   # sum_j max(tau*dq, 0)   [W]
        self.tau2 = torch.zeros(n, device=dev)    # mean_j tau^2
        self.nsub = 0

    def __call__(self, actions):
        env = self.env
        if self.k > 0:
            if self.hist is None:
                self.hist = [torch.zeros_like(actions) for _ in range(self.k + 1)]
            self.hist.append(actions.clone())
            self.hist.pop(0)
            actions = self.hist[0]
        tau = self._orig(actions)
        self.p_pos += torch.clamp(tau * env.dof_vel, min=0).sum(dim=1)
        self.tau2 += (tau ** 2).mean(dim=1)
        self.nsub += 1
        return tau


def set_friction(env, mu_eff):
    # PhysX menggabungkan gesekan dengan rata-rata; lantai diset 0 (lihat main),
    # jadi gesek efektif = mu_robot / 2.
    mu = 2.0 * mu_eff
    for e, a in zip(env.envs, env.actor_handles):
        props = env.gym.get_actor_rigid_shape_properties(e, a)
        for p in props:
            p.friction = mu
        env.gym.set_actor_rigid_shape_properties(e, a, props)


def set_payload(env, base_mass0, kg):
    for e, a in zip(env.envs, env.actor_handles):
        props = env.gym.get_actor_rigid_body_properties(e, a)
        props[0].mass = base_mass0 + kg
        env.gym.set_actor_rigid_body_properties(e, a, props, recomputeInertia=True)


def apply_condition(env, inst, base_mass0, base_noise, cond):
    env.cfg.domain_rand.push_robots = cond["push"] > 0
    env.cfg.domain_rand.max_push_vel_xy = float(cond["push"])
    set_friction(env, cond["friction"])
    set_payload(env, base_mass0, cond["payload"])
    env.noise_scale_vec = base_noise * cond["obs_noise"]
    env.add_noise = cond["obs_noise"] > 0
    sim_dt_ms = env.sim_params.dt * 1000.0
    inst.set_delay(round(cond["delay_ms"] / sim_dt_ms))


@torch.no_grad()
def run_repeat(env, policy, inst, T, total_mass):
    n = env.num_envs
    dev = env.device
    n_steps = int(round(T / env.dt))
    obs, _ = env.reset()
    inst.hist = None
    inst.clear()

    alive = torch.ones(n, dtype=torch.bool, device=dev)
    steps = torch.zeros(n, device=dev)
    acc = {k: torch.zeros(n, device=dev) for k in
           ["verr2", "werr2", "tilt", "tilt_max", "tiltrate2", "vz2", "ppos", "tau2",
            "dact", "height", "vx"]}
    prev_a = torch.zeros(n, env.num_actions, device=dev)
    x0 = env.root_states[:, 0:2].clone()

    for _ in range(n_steps):
        a = policy(obs.detach())
        obs, _, _, dones, _ = env.step(a.detach())
        m = alive & ~dones.bool()               # langkah valid: hidup sebelum & sesudah step
        mf = m.float()
        steps += mf

        verr = env.commands[:, :2] - env.base_lin_vel[:, :2]
        werr = env.commands[:, 2] - env.base_ang_vel[:, 2]
        tilt = torch.rad2deg(torch.acos(torch.clamp(-env.projected_gravity[:, 2], -1, 1)))
        acc["verr2"] += (verr ** 2).sum(1) * mf
        acc["werr2"] += werr ** 2 * mf
        acc["tilt"] += tilt * mf
        acc["tilt_max"] = torch.maximum(acc["tilt_max"], tilt * mf)
        acc["tiltrate2"] += (env.base_ang_vel[:, :2] ** 2).sum(1) * mf
        acc["vz2"] += env.base_lin_vel[:, 2] ** 2 * mf
        ns = max(inst.nsub, 1)
        acc["ppos"] += inst.p_pos / ns * mf
        acc["tau2"] += inst.tau2 / ns * mf
        acc["dact"] += (a - prev_a).abs().mean(1) * mf
        acc["height"] += env.root_states[:, 2] * mf
        acc["vx"] += env.base_lin_vel[:, 0] * mf
        prev_a = a.clone()
        inst.clear()
        alive &= ~dones.bool()
        if not alive.any():
            break

    s = steps.clamp(min=1)
    res = {
        "success": alive.float(),
        "time_alive_s": steps * env.dt,
        "vel_rmse": torch.sqrt(acc["verr2"] / s),
        "yaw_rmse": torch.sqrt(acc["werr2"] / s),
        "tilt_mean_deg": acc["tilt"] / s,
        "tilt_max_deg": acc["tilt_max"],
        "tiltrate_rms": torch.sqrt(acc["tiltrate2"] / s),
        "bounce_vz_rms": torch.sqrt(acc["vz2"] / s),
        "power_pos_W": acc["ppos"] / s,
        "torque_rms_Nm": torch.sqrt(acc["tau2"] / s),
        "dact_per_step": acc["dact"] / s,
        "base_height": acc["height"] / s,
        "vx_mean": acc["vx"] / s,
    }
    vx = res["vx_mean"].clamp(min=0.05)
    res["cot"] = res["power_pos_W"] / (total_mass * 9.81 * vx)
    return {k: v.cpu().numpy() for k, v in res.items()}


def benchmark_policy(jit_path, iters, obs_dim, control_hz):
    out = {}
    model = torch.jit.load(jit_path, map_location="cpu").eval()
    torch.set_num_threads(1)
    x = torch.randn(1, obs_dim)
    with torch.no_grad():
        for _ in range(200):
            model(x)
        t = np.empty(iters)
        for i in range(iters):
            t0 = time.perf_counter()
            model(x)
            t[i] = (time.perf_counter() - t0) * 1e3
    out["cpu1_ms_mean"], out["cpu1_ms_p99"] = float(t.mean()), float(np.percentile(t, 99))
    if torch.cuda.is_available():
        mg = model.to("cuda")
        xg = x.cuda()
        with torch.no_grad():
            for _ in range(200):
                mg(xg)
            torch.cuda.synchronize()
            for i in range(iters):
                t0 = time.perf_counter()
                mg(xg)
                torch.cuda.synchronize()
                t[i] = (time.perf_counter() - t0) * 1e3
        out["gpu_ms_mean"] = float(t.mean())
    out["inferences_per_s"] = control_hz
    out["cpu_load_pct"] = out["cpu1_ms_mean"] * 1e-3 * control_hz * 100.0
    return out


def main():
    args = get_args()
    n_envs = args.num_envs or 1000
    args.num_envs = n_envs
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = n_envs
    env_cfg.env.episode_length_s = _own.episode_s + 5.0   # jangan timeout sebelum evaluasi selesai
    env_cfg.env.test = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.static_friction = 0.0     # gesek efektif diatur lewat gesek robot (set_friction)
    env_cfg.terrain.dynamic_friction = 0.0
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    # perintah tetap, sama untuk semua env dan semua frekuensi
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

    exp = train_cfg.runner.experiment_name
    jit_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", exp, "exported", "policies")
    export_policy_as_jit(runner.alg.actor_critic, jit_dir)
    jit_path = os.path.join(jit_dir, "policy_1.pt")

    inst = Instrument(env)
    # Mengubah properti body sebelum simulasi pernah dijalankan merusak run pertama
    # (robot jatuh <1 s). Jalankan fisika dulu sebelum menyentuh properti apa pun.
    run_repeat(env, policy, inst, 2.0, 20.0)

    props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])
    total_mass = float(sum(p.mass for p in props))
    base_mass0 = float(props[0].mass)
    base_noise = env.noise_scale_vec.clone()

    control_hz = 1.0 / env.dt
    meta = dict(task=args.task, decimation=env_cfg.control.decimation, sim_dt=env.sim_params.dt,
                policy_dt=env.dt, control_hz=control_hz, total_mass_kg=total_mass,
                num_envs=n_envs, repeats=_own.repeats, episode_s=_own.episode_s, vx_cmd=_own.vx,
                nominal=NOMINAL, sweeps=SWEEPS)
    meta["compute"] = benchmark_policy(jit_path, _own.bench_iters, env.num_obs, control_hz)
    print("[compute]", meta["compute"])

    jobs = []
    for ax in _own.axes.split(","):
        ax = ax.strip()
        if ax == "nominal":
            jobs.append(("nominal", 0.0, dict(NOMINAL)))
        else:
            for lv in SWEEPS[ax]:
                c = dict(NOMINAL)
                c[ax] = lv
                jobs.append((ax, lv, c))

    os.makedirs(_own.out, exist_ok=True)
    rows = []
    t_start = time.time()
    for ji, (ax, lv, cond) in enumerate(jobs):
        apply_condition(env, inst, base_mass0, base_noise, cond)
        run_repeat(env, policy, inst, 1.5, total_mass + cond["payload"])   # settle, dibuang
        for rep in range(_own.repeats):
            res = run_repeat(env, policy, inst, _own.episode_s, total_mass + cond["payload"])
            row = dict(task=args.task, decimation=meta["decimation"], control_hz=control_hz,
                       axis=ax, level=lv, repeat=rep, n_envs=n_envs)
            for k, v in res.items():
                row[k] = float(v.mean())
                row[k + "_std"] = float(v.std())
            rows.append(row)
        last = rows[-1]
        print(f"[{ji+1}/{len(jobs)}] {ax}={lv}  success={last['success']:.3f} "
              f"t_alive={last['time_alive_s']:.1f}s vel_rmse={last['vel_rmse']:.3f} "
              f"tilt={last['tilt_mean_deg']:.2f}deg cot={last['cot']:.2f}  "
              f"({time.time()-t_start:.0f}s)", flush=True)

    csv_path = os.path.join(_own.out, f"{args.task}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(_own.out, f"{args.task}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved", csv_path)


if __name__ == "__main__":
    main()
