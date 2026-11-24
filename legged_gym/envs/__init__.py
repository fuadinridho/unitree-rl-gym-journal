from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.g1.g1_decimation_v2 import DECIMATIONS, SEEDS, make_cfgs, task_name
from legged_gym.envs.g1.g1_env import G1Robot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

# policy rate = 200 / decimation Hz; seed 0 has no suffix, seed K is "_sK"
for _dec in DECIMATIONS:
    for _seed in SEEDS:
        _cfg, _ppo = make_cfgs(_dec, _seed)
        task_registry.register(task_name(_dec, _seed), G1Robot, _cfg(), _ppo())
