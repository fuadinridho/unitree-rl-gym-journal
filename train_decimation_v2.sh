#!/bin/bash
# Train g1_decimation*_v2 (hyperparameter konsisten terhadap waktu), dibagi ke beberapa GPU.
#
# Usage:
#   ./train_decimation_v2.sh                       # tingkat 1+2, seed 0: dec 1 2 3 5 + 4 8 12 16 20 24 28 32 36 40
#   ./train_decimation_v2.sh -t 1                  # hanya tingkat 1 (dec 1 2 3 5)
#   ./train_decimation_v2.sh -g "0 1" -j 3         # GPU 0 dan 1, 3 job paralel per GPU
#   ./train_decimation_v2.sh -s "1 2" 3 4 5 8 12 16 24   # tingkat 3: seed 1 dan 2 untuk dec tertentu
# Env: EXTRA_ARGS="--max_iterations 300" (diteruskan ke train.py)
# Task: g1_decimation{N}_v2 (seed 0) atau g1_decimation{N}_v2_s{K}; log: logs/sweep_stdout_v2/

GPUS="0 1"
JOBS=3
SEEDS="0"
TIER=12
while getopts "g:j:s:t:" o; do
    case $o in g) GPUS=$OPTARG;; j) JOBS=$OPTARG;; s) SEEDS=$OPTARG;; t) TIER=$OPTARG;; esac
done
shift $((OPTIND-1))
DECS=("$@")
if [ ${#DECS[@]} -eq 0 ]; then
    case $TIER in
        1)  DECS=(1 2 3 5);;
        2)  DECS=(4 8 12 16 20 24 28 32 36 40);;
        *)  DECS=(1 2 3 5 4 8 12 16 20 24 28 32 36 40);;
    esac
fi

cd "$(dirname "$0")"
LOGDIR=logs/sweep_stdout_v2
mkdir -p "$LOGDIR"
GPU_ARR=($GPUS); NG=${#GPU_ARR[@]}

TASKS=()
for s in $SEEDS; do
    for d in "${DECS[@]}"; do
        if [ "$s" = "0" ]; then TASKS+=("g1_decimation${d}_v2"); else TASKS+=("g1_decimation${d}_v2_s${s}"); fi
    done
done
echo "Tasks (${#TASKS[@]}): ${TASKS[*]} | GPU: $GPUS | parallel/GPU: $JOBS"

for gi in "${!GPU_ARR[@]}"; do
    gpu=${GPU_ARR[$gi]}
    mine=()
    for i in "${!TASKS[@]}"; do [ $((i % NG)) -eq "$gi" ] && mine+=("${TASKS[$i]}"); done
    [ ${#mine[@]} -eq 0 ] && continue
    (
        printf '%s\n' "${mine[@]}" | xargs -P "$JOBS" -I{} bash -c '
            echo "[$(date +%H:%M:%S)] START {} (GPU '"$gpu"')"
            CUDA_VISIBLE_DEVICES='"$gpu"' python legged_gym/scripts/train.py \
                --task={} --headless '"$EXTRA_ARGS"' > '"$LOGDIR"'/{}.log 2>&1
            echo "[$(date +%H:%M:%S)] DONE  {} (exit $?)"; sleep 20'
    ) &
    sleep 20
done
wait
echo "All done."
