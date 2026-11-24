"""Foot trajectories (substep resolution, 200 Hz) of trained v2 policies at a fixed command.

Records, for every env that does not fall: pelvis position/yaw, both foot positions and contact flags.
Output: <out>/<task>_foot.npz  (t, root[n,3], yaw[n], foot[n,2,3], contact[n,2], alive[n])

usage:
  CUDA_VISIBLE_DEVICES=1 python legged_gym/scripts/foot_trajectory.py --task=g1_decimation40_v2 --headless --num_envs 8
"""
import argparse
import os

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--out", default="eval_results_v2_energy/foot")
_p.add_argument("--episode_s", type=float, default=20.0)
_p.add_argument("--vx", type=float, default=0.6)
_own, _rest = _p.parse_known_args()
import sys  # noqa: E402
sys.argv = [sys.argv[0]] + _rest

import isaacgym  # noqa: E402,F401
from legged_gym.envs import *  # noqa: E402,F401,F403
import numpy as np  # noqa: E402
import torch  # noqa: E402
from legged_gym.utils import get_args, task_registry  # noqa: E402

FORCE_THR = 1.0


class FootRecorder:
    """Wraps env._compute_torques; stores foot kinematics at every physics substep."""

    def __init__(self, env):
        self.env = env
        self._orig = env._compute_torques
        env._compute_torques = self
        self.buf = []
        self.on = False

    def __call__(self, actions):
        env = self.env
        if self.on:
            env.gym.refresh_net_contact_force_tensor(env.sim)
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            fz = env.contact_forces[:, env.feet_indices, 2]
            # read the live body tensor: env.feet_pos / env.root_states are only refreshed once per policy step
            rb = env.rigid_body_states_view
            q = rb[:, 0, 3:7]
            yaw = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                              1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
            self.buf.append(torch.cat([rb[:, 0, :3], yaw[:, None],
                                       rb[:, env.feet_indices, :3].reshape(env.num_envs, 6),
                                       (fz > FORCE_THR).float()], dim=1).cpu())
        return self._orig(actions)


def main():
    args = get_args()
    n_envs = args.num_envs or 32
    args.num_envs = n_envs
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = n_envs
    env_cfg.env.episode_length_s = _own.episode_s + 10.0
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
    rec = FootRecorder(env)

    obs, _ = env.reset()
    alive = torch.ones(n_envs, dtype=torch.bool, device=env.device)
    warm = int(round(2.0 / env.dt))
    n_steps = int(round(_own.episode_s / env.dt))
    with torch.no_grad():
        for i in range(warm + n_steps):
            rec.on = i >= warm
            obs, _, _, dones, _ = env.step(policy(obs.detach()).detach())
            alive &= ~dones.bool()
    arr = torch.stack(rec.buf).numpy()               # [T, n, 12]
    os.makedirs(_own.out, exist_ok=True)
    f = os.path.join(_own.out, f"{args.task}_foot.npz")
    np.savez(f, dt=env.sim_params.dt, control_hz=1.0 / env.dt, root=arr[:, :, 0:3], yaw=arr[:, :, 3],
             foot=arr[:, :, 4:10].reshape(arr.shape[0], n_envs, 2, 3), contact=arr[:, :, 10:12],
             alive=alive.cpu().numpy())
    print("saved", f, arr.shape, "alive", int(alive.sum()), "/", n_envs)


if __name__ == "__main__":
    main()
