"""G1 decimation sweep v2: hyperparameter RL konsisten terhadap waktu.

Beda dengan g1_decimation{N} (v1), di sini yang berubah hanya frekuensi kontrol:
  - gamma, lam      : dipetakan ke cakrawala reward yang sama (0.99 / 0.95 per 20 ms)
  - num_steps_per_env: rollout >= 0.48 s (24 langkah pada 50 Hz)
  - dof_acc         : dibagi dt referensi 0.02 s, bukan dt kontrol
Task: g1_decimation{N}_v2 (seed 0) dan g1_decimation{N}_v2_s{K} (seed K).
"""
from legged_gym.envs.g1.g1_decimation4 import G1Decimation4Cfg, G1Decimation4CfgPPO

SIM_DT = 0.005
REF_DT = 0.02
DECIMATIONS = [1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 28, 32, 36, 40]
SEEDS = [0, 1, 2]


def task_name(dec, seed=0):
    return f"g1_decimation{dec}_v2" + (f"_s{seed}" if seed else "")


def make_cfgs(dec, seed=0):
    dt = SIM_DT * dec
    ratio = dt / REF_DT
    hz = 1.0 / dt
    name = task_name(dec, seed)

    class Cfg(G1Decimation4Cfg):
        __doc__ = f"G1 decimation {dec} v2: sim dt {SIM_DT} s, control {hz:g} Hz (policy dt {dt:g} s)"

        class control(G1Decimation4Cfg.control):
            decimation = dec

        class rewards(G1Decimation4Cfg.rewards):
            dof_acc_ref_dt = REF_DT

    class PPO(G1Decimation4CfgPPO):
        seed = None

        class algorithm(G1Decimation4CfgPPO.algorithm):
            gamma = 0.99 ** ratio
            lam = 0.95 ** ratio

        class runner(G1Decimation4CfgPPO.runner):
            experiment_name = name
            num_steps_per_env = max(24, round(0.48 / dt))

    PPO.seed = seed
    Cfg.__name__ = f"G1Decimation{dec}V2Cfg"
    PPO.__name__ = f"G1Decimation{dec}V2CfgPPO"
    return Cfg, PPO
