#!/bin/bash
# Evaluasi policy g1_decimation*_v2 (Isaac Gym), dibagi ke beberapa GPU. Hasil: eval_results_v2/
#
# Usage:
#   ./run_eval_decimation_v2.sh                         # tingkat 1+2, seed 0
#   ./run_eval_decimation_v2.sh -s "0 1 2" 3 4 5 8      # beberapa seed untuk dec tertentu
#   ./run_eval_decimation_v2.sh -g "1" -j 2             # hanya GPU 1, 2 job paralel
# Env: EXTRA_ARGS="--num_envs 200 --repeats 1 --episode_s 10" untuk tes cepat

GPUS="0 1"; JOBS=1; SEEDS="0"
while getopts "g:j:s:" o; do
    case $o in g) GPUS=$OPTARG;; j) JOBS=$OPTARG;; s) SEEDS=$OPTARG;; esac
done
shift $((OPTIND-1))
DECS=("$@")
[ ${#DECS[@]} -eq 0 ] && DECS=(1 2 3 4 5 8 12 16 20 24 28 32 36 40)

cd "$(dirname "$0")"
OUT=${OUT:-eval_results_v2}
mkdir -p logs/eval_stdout_v2 "$OUT"
GPU_ARR=($GPUS); NG=${#GPU_ARR[@]}
TASKS=()
for s in $SEEDS; do for d in "${DECS[@]}"; do
    if [ "$s" = "0" ]; then TASKS+=("g1_decimation${d}_v2"); else TASKS+=("g1_decimation${d}_v2_s${s}"); fi
done; done

for gi in "${!GPU_ARR[@]}"; do
    gpu=${GPU_ARR[$gi]}
    mine=()
    for i in "${!TASKS[@]}"; do [ $((i % NG)) -eq "$gi" ] && mine+=("${TASKS[$i]}"); done
    [ ${#mine[@]} -eq 0 ] && continue
    echo "GPU $gpu: ${mine[*]} (parallel $JOBS)"
    (
        printf '%s\n' "${mine[@]}" | xargs -P "$JOBS" -I{} bash -c '
            echo "[$(date +%H:%M:%S)] START {} on GPU '"$gpu"'"
            CUDA_VISIBLE_DEVICES='"$gpu"' python legged_gym/scripts/eval_decimation.py \
                --task={} --headless --out '"$OUT"' '"$EXTRA_ARGS"' \
                > logs/eval_stdout_v2/{}.log 2>&1
            echo "[$(date +%H:%M:%S)] DONE  {} (exit $?)"'
    ) &
    sleep 15
done
wait
echo "Eval selesai. Hasil di $OUT/"
