# Policy rate and energy cost of RL humanoid walking (Unitree G1)

Code, trained policies and data for a controlled sweep of the policy rate, 5 to 200 Hz, on the 12-DoF Unitree G1 in
Isaac Gym. Physics and the PD loop run at 200 Hz; the policy rate is `200 / decimation` Hz.

Based on [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) (BSD-3-Clause, see `LICENSE`). Only the
files needed for this study are included.

## Contents

| Path | Content |
|---|---|
| `legged_gym/envs/g1/g1_decimation_v2.py` | Task definitions: what is rescaled with the policy rate |
| `legged_gym/envs/g1/g1_decimation4.py` | Base G1 configuration (PD gains, rewards, commands) |
| `legged_gym/scripts/` | Training, evaluation and analysis scripts |
| `logs/<task>/<run>/model_10000.pt` | Final checkpoint of the 30 policies |
| `eval_results_v2/` | Robustness evaluation (push, payload, friction, delay, noise) and `ranking.csv` |
| `eval_results_v2_energy/` | Per-environment energy and support-phase data, summaries, statistics, figures |
| `resources/robots/g1_description/` | G1 12-DoF URDF and meshes |

Tasks are `g1_decimation{D}_v2` (seed 0) and `g1_decimation{D}_v2_s{K}` (seed K). D = 2, 3, 4, 5, 8, 12, 16, 24 have
three seeds; D = 1, 20, 28, 32, 36, 40 have one.

## Setup

Python 3.8, PyTorch, Isaac Gym Preview 4, [rsl_rl v1.0.2](https://github.com/leggedrobotics/rsl_rl/tree/v1.0.2),
pandas, scipy, matplotlib. The results were produced with PyTorch 2.3.1 and NumPy 1.24.4. Then `pip install -e .`

## Reproduce

Run from the repository root.

```bash
# tables and figures from the included data (no simulator needed)
python legged_gym/scripts/energy_gait_stats.py
python legged_gym/scripts/rank_decimation.py

# regenerate the data from the included policies
python legged_gym/scripts/energy_gait_analysis.py --task=g1_decimation8_v2 --headless --num_envs 1000
python legged_gym/scripts/foot_trajectory.py      --task=g1_decimation8_v2 --headless --num_envs 32   # all 14 rates, then:
python legged_gym/scripts/foot_trajectory_plot.py
python legged_gym/scripts/eval_decimation.py      --task=g1_decimation8_v2 --headless

# train
python legged_gym/scripts/train.py --task=g1_decimation8_v2 --headless
```

## Notes on the metrics

- Distance is measured along the pelvis path (sum of displacements over 0.8 s intervals), not along a fixed world axis,
  because the heading is not controlled and drifts. `dist_world_x_m` is kept for comparison.
- A step is counted after at least 50 ms without contact, which rejects contact chatter. Support phases use a 1 N
  threshold at every 5 ms physics step.
- Rates below 8 Hz have one seed each and are excluded from the main statistics (`sensitivity_all_rates` in
  `stats.json` includes them).
- Raw foot recordings (about 80 MB) are not included; `foot_stats.csv` holds the summary.
- All results are from simulation. There is no hardware validation.
