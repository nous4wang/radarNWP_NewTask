#!/usr/bin/env bash
# The whole matrix under the score-aware objective.
#
# The decisive control showed that one model reaches 0.118 at 2 mm when
# only the objective changes. That is a statement about one model, and
# the paper's claim is about the family, so the family has to be run.
#
# The matrix is the eighteen models of the earlier comparison, every one
# of them trained with the score-aware objective instead of the
# benchmark's log-space regression. The point is not which model wins --
# it is what the scores become when the objective stops suppressing the
# heavy rain.
#
# Eight cards, the models in order of how long they take, so that the
# short ones finish first and the cards do not idle at the end.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/score_matrix}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
LR="${LR:-3e-4}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

TIERS="0.1 0.5 2.0 4.0"

mkdir -p "${OUT_DIR}"

# The models, in the order they are dispatched.
MODELS=(
    "simvp_v2"
    "simvp_v2_nwp_concat"
    "simvp_v2_quantum"
    "simvp_v2_nwp_xattn"
    "simvp_v2_nwp_delta"
    "simvp_early_concat"
    "simvp_radar_only"
    "convlstm"
    "predrnn"
    "earthformer"
    "metnet3"
    "diffcast"
    "cascast_nwp"
    "mocast"
    "pw_foucast"
    "fusioncast"
    "mmf_rnn"
    "furecast"
)

# The training is dispatched one at a time onto whichever card is free.
# The analytical baseline has no weights to fit, so it is evaluated
# rather than trained.
declare -A PENDING
for model in "${MODELS[@]}"; do
    PENDING["${model}"]=1
done

gpu_is_free() {
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i "${1}" 2>/dev/null | head -1 || echo 999999)"
    [[ -z "${used}" ]] && return 1
    (( used < 100 ))
}

echo "=== phase 1: training, one model per free card ==="
remaining="${#MODELS[@]}"
while (( remaining > 0 )); do
    for gpu in ${GPUS}; do
        if ! gpu_is_free "${gpu}"; then
            continue
        fi
        # Take the next model that has not started.
        picked=""
        for model in "${MODELS[@]}"; do
            if [[ -n "${PENDING[${model}]:-}" && "${PENDING[${model}]}" == "1" ]]; then
                picked="${model}"
                break
            fi
        done
        [[ -z "${picked}" ]] && break

        PENDING["${picked}"]=0
        rundir="${OUT_DIR}/${picked}"
        log="${OUT_DIR}/${picked}.log"
        mkdir -p "${rundir}"
        echo "  -> ${picked} on GPU ${gpu}"
        (
            CUDA_VISIBLE_DEVICES="${gpu}" \
            OMP_NUM_THREADS=4 \
            PYTHONUNBUFFERED=1 \
            "${PYTHON}" "${REPO_ROOT}/scripts/train.py" \
                --model "${picked}" \
                --data-root "${DATA_ROOT}" \
                --packed \
                --split train \
                --epochs "${EPOCHS}" \
                --batch-size "${BATCH_SIZE}" \
                --num-workers "${WORKERS}" \
                --patch-size "${PATCH_SIZE}" \
                --slide-stride "${SLIDE_STRIDE}" \
                --lr "${LR}" \
                --loss balanced \
                --alpha-csi 0.5 \
                --tier-thresholds ${TIERS} \
                --log-every 400 \
                --device cuda \
                --out-dir "${rundir}" \
                > "${log}" 2>&1
            echo "  [${picked}] $(grep -E '^epoch [0-9]+:' "${log}" | tail -1)"
        ) &
        sleep 6
        remaining=$((remaining - 1))
    done
    sleep 20
done
wait
echo "all training finished"

echo
echo "=== phase 2: evaluation (clean test split) ==="
for model in "${MODELS[@]}"; do
    ckpt="$(ls -dt "${OUT_DIR}/${model}"/*/last.pt 2>/dev/null | head -1 || true)"
    if [[ -z "${ckpt}" ]]; then
        echo "  [${model}] no checkpoint -- skipped"
        continue
    fi
    (
        CUDA_VISIBLE_DEVICES=0 \
        "${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
            --model "${model}" \
            --checkpoint "${ckpt}" \
            --data-root "${DATA_ROOT}" \
            --packed \
            --split test \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${WORKERS}" \
            --patch-size "${PATCH_SIZE}" \
            --slide-stride "${SLIDE_STRIDE}" \
            --thresholds ${TIERS} \
            --device cuda \
            --out-json "${OUT_DIR}/${model}.eval.json" \
            > "${OUT_DIR}/${model}.eval.log" 2>&1
        echo "  [${model}] evaluated"
    ) &
    sleep 3
done
wait

echo
echo "=== the summary ==="
"${PYTHON}" - "${OUT_DIR}" <<'PYEOF'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])

# The same models under the benchmark's own objective, for the contrast.
OLD = Path("/home/wf/radarNWP/runs/hqma")


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def row(name, report):
    t = report["thresholds"]

    def g(thr, key):
        return t.get(thr, {}).get(key, float("nan"))

    return (
        name,
        g("0.1", "csi"), g("0.5", "csi"), g("2.0", "csi"), g("4.0", "csi"),
        g("0.5", "far"), g("0.5", "bias"), g("2.0", "hits"),
    )


rows = []
for path in sorted(out.glob("*.eval.json")):
    report = read(path)
    if report is not None:
        rows.append(row(path.stem.replace(".eval", ""), report))

if rows:
    print(f"{'model (score-aware objective)':40s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}"
          f"{'CSI@2.0':>9s}{'CSI@4.0':>9s}{'FAR@0.5':>9s}{'BIAS@0.5':>9s}"
          f"{'hits@2.0':>12s}")
    print("-" * 106)
    for r in rows:
        print(f"{r[0]:40s}" + "".join(f"{v:9.3f}" for v in r[1:7])
              + f"{int(r[7]):12d}")

print()
print("The contrast, for the models that were run both ways:")
print(f"{'model':28s}{'CSI@2.0 old':>13s}{'CSI@2.0 new':>13s}"
      f"{'CSI@4.0 old':>13s}{'CSI@4.0 new':>13s}")
for path in sorted(out.glob("*.eval.json")):
    name = path.stem.replace(".eval", "")
    old = read(OLD / f"{name}.eval.json")
    new = read(path)
    if old is None or new is None:
        continue
    ot, nt = old["thresholds"], new["thresholds"]
    print(f"{name:28s}"
          f"{ot.get('2.0', {}).get('csi', float('nan')):13.3f}"
          f"{nt.get('2.0', {}).get('csi', float('nan')):13.3f}"
          f"{ot.get('4.0', {}).get('csi', float('nan')):13.3f}"
          f"{nt.get('4.0', {}).get('csi', float('nan')):13.3f}")
PYEOF

echo
echo "=== done: reports in ${OUT_DIR} ==="
