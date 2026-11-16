#!/usr/bin/env bash
# The spectral diagnostic for the whole matrix.
#
# The claim that every method on this benchmark blurs the convective cores
# away rests on one model at the moment. A claim about a family needs the
# family: the script runs the diagnostic for every checkpoint and prints
# the two band ratios next to each other, so the spread is visible.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs/tgrs}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/rapsd}"
MAX_CASES="${MAX_CASES:-4}"
GPUS="${GPUS:-4 5 6 7}"

read -r -a CARDS <<< "${GPUS}"

MODELS=(
    pysteps_adv convlstm predrnn simvp_v2 earthformer
    simvp_v2_quantum simvp_v2_nwp_concat simvp_v2_nwp_xattn simvp_v2_nwp_delta
    metnet3 diffcast cascast_nwp mocast pw_foucast fusioncast mmf_rnn furecast
    simvp_radar_only
)

mkdir -p "${OUT_DIR}"

i=0
for model in "${MODELS[@]}"; do
    ckpt="$(ls -dt "${RUNS_DIR}/${model}"/*/last.pt 2>/dev/null | head -1 || true)"
    if [[ -z "${ckpt}" ]]; then
        echo "  [${model}] no checkpoint -- skipped"
        continue
    fi
    if [[ -f "${OUT_DIR}/${model}/rapsd.json" ]]; then
        echo "  [${model}] already done -- skipped"
        continue
    fi
    gpu="${CARDS[$(( i % ${#CARDS[@]} ))]}"
    (
        mkdir -p "${OUT_DIR}/${model}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        "${PYTHON}" "${REPO_ROOT}/scripts/rapsd.py" \
            --data-root "${DATA_ROOT}" \
            --packed \
            --split test \
            --model "${model}" \
            --checkpoint "${ckpt}" \
            --out-dir "${OUT_DIR}/${model}" \
            --max-cases "${MAX_CASES}" \
            --device cuda \
            > "${OUT_DIR}/${model}.log" 2>&1
        echo "  [${model}] done"
    ) &
    i=$(( i + 1 ))
    sleep 3
done
wait

echo
echo "=== the two band ratios for every model ==="
"${PYTHON}" - <<PYEOF
import json
from pathlib import Path

out = Path("${OUT_DIR}")
rows = []
for directory in sorted(out.iterdir()):
    path = directory / "rapsd.json"
    if not path.exists():
        continue
    r = json.loads(path.read_text())
    s = r.get("summary", {})
    rows.append((
        directory.name,
        s.get("high_k_ratio", float("nan")),
        s.get("low_k_ratio", float("nan")),
        r.get("n_windows", r.get("n_samples", "?")),
    ))
rows.sort(key=lambda row: row[1] if row[1] == row[1] else 1e9)
print(f"{'model':26s}{'high_k (<10km)':>17s}{'low_k (>30km)':>15s}{'windows':>10s}")
print("-" * 68)
for name, high, low, n in rows:
    print(f"{name:26s}{high:17.4f}{low:15.4f}{str(n):>10s}")
print()
print("a high_k ratio below one means the forecast lost the small-scale energy")
print("(the convective cores were blurred away); above one means it added")
print("some (a noise floor).")
PYEOF
