#!/bin/bash
# usage: ./run_energy_analysis.sh GPU task1 task2 ...
gpu=$1; shift
mkdir -p logs/energy_stdout
for t in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu python legged_gym/scripts/energy_gait_analysis.py --task=$t --headless --num_envs 1000 --repeats 2 --out eval_results_v2_energy > logs/energy_stdout/$t.log 2>&1
done
