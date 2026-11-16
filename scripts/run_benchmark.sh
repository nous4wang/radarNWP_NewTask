#!/usr/bin/env bash
#
# Train and evaluate every registered baseline on the radar-NWP benchmark.
#
# Runs, for each of the eight adapters:
#     scripts/train.py    -> runs/<model>/<model>_<timestamp>/{last.pt,history.json}
#     scripts/evaluate.py -> runs/<model>/eval_report.json
#
# Works unchanged under Git Bash on Windows and on a Linux server.
#
# Usage:
#     bash scripts/run_benchmark.sh
#     bash scripts/run_benchmark.sh --models mocast,diffcast
#     DATA_ROOT=/data/BJ PACKED=1 EPOCHS=20 bash scripts/run_benchmark.sh
#
# Every setting is an environment variable with a default; see --help.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------- defaults --

# Root of the dataset. Prefer a repo-local data/BJ; fall back to the archive
# on this workstation when it is present.
DEFAULT_DATA_ROOT="data/BJ"
if [ ! -d "${DEFAULT_DATA_ROOT}" ] && [ -d "R:/1radarNWPprediction/data/BJ" ]; then
  DEFAULT_DATA_ROOT="R:/1radarNWPprediction/data/BJ"
fi

DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
PACKED="${PACKED:-0}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PATCH_SIZE="${PATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
ALIGNMENT="${ALIGNMENT:-forecast}"
DEVICE="${DEVICE:-auto}"
RUNS_DIR="${RUNS_DIR:-runs}"
SEED="${SEED:-0}"
IN_LEN="${IN_LEN:-11}"
OUT_LEN="${OUT_LEN:-10}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-${NUM_WORKERS}}"
# 0 means "no limit" for both --max-steps and --max-batches.
MAX_STEPS="${MAX_STEPS:-0}"
MAX_BATCHES="${MAX_BATCHES:-0}"
PYTHON="${PYTHON:-python}"

# The model list is discovered from the registry rather than hard-coded, so a
# new baseline is picked up by running this script without editing it.
# The registry discovers models/<name>/model.py lazily by path.
discover_models() {
  "${PYTHON}" - <<'PY'
import sys

sys.path.insert(0, ".")
try:
    from radar_nwp_bench.models import list_models
except Exception as exc:  # pragma: no cover - discovery must not abort the run
    print(f"# model discovery failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(" ".join(list_models()))
PY
}

ALL_MODELS="$(discover_models || true)"
if [ -z "${ALL_MODELS}" ]; then
  echo "error: could not discover the registered baselines." >&2
  echo "       run 'python -c \"from radar_nwp_bench.models import list_models; print(list_models())\"'" >&2
  exit 1
fi
MODELS="${ALL_MODELS}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_benchmark.sh [--models m1,m2,...]

Trains and evaluates every registered baseline, writing results under
runs/<model>/.

Options:
  --models LIST   Comma-separated subset of baselines to run.
                  Default: every baseline registered in the model registry
                  (discovered at run time, so a newly added baseline is
                  picked up automatically).
  -h, --help      Show this message and exit.

Environment overrides (all optional):
  DATA_ROOT        Dataset root: raw CP_* tree, or packed .npz dir.
                   Default: data/BJ, else the local archive when present.
  PACKED           1/true to read .npz containers. Default: 0
  EPOCHS           Training epochs. Default: 10
  BATCH_SIZE       Training batch size. Default: 2
  PATCH_SIZE       Patch side in pixels. Default: 128
  NUM_WORKERS      Dataloader workers. Default: 2
  ALIGNMENT        NWP alignment policy. Default: forecast
                   (bracket|forecast|analysis_pair|full_series)
  DEVICE           Torch device. Default: auto
  RUNS_DIR         Output root. Default: runs
  SEED             Random seed. Default: 0
  IN_LEN           Radar history frames. Default: 11
  OUT_LEN          Forecast frames. Default: 10
  LR               Learning rate. Default: 2e-4
  WEIGHT_DECAY     Weight decay. Default: 1e-4
  EVAL_BATCH_SIZE  Evaluation batch size. Default: BATCH_SIZE
  EVAL_NUM_WORKERS Evaluation workers. Default: NUM_WORKERS
  MAX_STEPS        Cap training steps (0 = no limit). Default: 0
  MAX_BATCHES      Cap evaluation batches (0 = no limit). Default: 0
  PYTHON           Python interpreter. Default: python
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --models)
      if [ "$#" -lt 2 ]; then
        echo "error: --models needs a value" >&2
        exit 2
      fi
      MODELS="$(printf '%s' "${2}" | tr ',' ' ')"
      shift 2
      ;;
    --models=*)
      MODELS="$(printf '%s' "${1#--models=}" | tr ',' ' ')"
      shift
      ;;
    *)
      echo "error: unknown argument '${1}'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "error: python interpreter '${PYTHON}' not found (override with PYTHON=...)" >&2
  exit 1
fi

if [ ! -d "${DATA_ROOT}" ]; then
  echo "error: DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  echo "       set DATA_ROOT to the raw archive or the packed .npz directory" >&2
  exit 1
fi

# `--packed` is a store_true flag; empty expansion drops the argument.
PACKED_FLAG=""
case "$(printf '%s' "${PACKED}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on) PACKED_FLAG="--packed" ;;
esac

mkdir -p "${RUNS_DIR}"
SUMMARY_FILE="${RUNS_DIR}/summary.tsv"
: > "${SUMMARY_FILE}"

# ------------------------------------------------------------------ helpers --

banner() {
  printf '\n'
  printf '================================================================\n'
  printf '  MODEL: %s\n' "$1"
  printf '================================================================\n'
}

# Read the metrics out of an evaluate.py report. Prints
# "csi@0.5 csi@2.0 ets@0.5 fss@0.5 dry-reference@0.5".
read_report() {
  "${PYTHON}" - "$1" <<'PY'
import json
import sys

NA = "n/a"


def main() -> None:
    try:
        with open(sys.argv[1], encoding="utf-8") as handle:
            report = json.load(handle)
    except Exception:
        print(NA, NA, NA, NA, NA)
        return

    thresholds = report.get("thresholds", {})
    reference = report.get("dry_reference_csi", {})

    def score(key: str, field: str) -> str:
        value = thresholds.get(key, {}).get(field)
        if value is None:
            return NA
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return NA

    try:
        base = f"{float(reference.get('0.5')):.3f}"
    except (TypeError, ValueError):
        base = NA

    print(
        score("0.5", "csi"),
        score("2.0", "csi"),
        score("0.5", "ets"),
        score("0.5", "fss"),
        base,
    )


main()
PY
}

# Fallback metric extraction from the table evaluate.py prints, used when the
# evaluation aborted before it could write its JSON report.
read_log_table() {
  awk '
    $2 == "|" && $1 ~ /^[0-9]+\.[0-9]+$/ { csi[$1] = $3; ets[$1] = $4; fss[$1] = $7 }
    $2 == "mm" && $3 == ":"               { pcsi[$1] = $5; pper[$1] = $7 }
    END {
      print ((("0.50" in csi) ? csi["0.50"] : "n/a"), \
            (("2.00" in csi) ? csi["2.00"] : "n/a"), \
            (("0.50" in ets) ? ets["0.50"] : "n/a"), \
            (("0.50" in fss) ? fss["0.50"] : "n/a"), \
            (("0.50" in pper) ? pper["0.50"] : "n/a"))
    }
  ' "$1"
}

# Persist the five headline numbers next to the run artifacts.
write_metrics() {
  local model_dir="$1" status="$2" metrics="$3"
  # shellcheck disable=SC2086
  set -- ${metrics}
  {
    printf '{\n'
    printf '  "status": "%s",\n' "${status}"
    printf '  "csi@0.5": "%s",\n' "${1:-n/a}"
    printf '  "csi@2.0": "%s",\n' "${2:-n/a}"
    printf '  "ets@0.5": "%s",\n' "${3:-n/a}"
    printf '  "fss@0.5": "%s",\n' "${4:-n/a}"
    printf '  "dry_reference_csi@0.5": "%s"\n' "${5:-n/a}"
    printf '}\n'
  } > "${model_dir}/eval_metrics.json"
}

# ------------------------------------------------------------------- runner --

printf 'radar-nwp-bench :: benchmark run\n'
printf 'data root : %s%s\n' "${DATA_ROOT}" "${PACKED_FLAG:+ (packed)}"
printf 'patch=%s epochs=%s batch=%s workers=%s alignment=%s device=%s\n' \
  "${PATCH_SIZE}" "${EPOCHS}" "${BATCH_SIZE}" "${NUM_WORKERS}" \
  "${ALIGNMENT}" "${DEVICE}"
printf 'outputs   : %s/<model>/\n' "${RUNS_DIR}"

for model in ${MODELS}; do
  banner "${model}"

  model_dir="${RUNS_DIR}/${model}"
  mkdir -p "${model_dir}"
  train_log="${model_dir}/train.log"
  eval_log="${model_dir}/eval.log"
  status="ok"

  # Unquoted ${PACKED_FLAG} on purpose: it expands to nothing when empty.
  if ! "${PYTHON}" scripts/train.py \
        --model "${model}" \
        --data-root "${DATA_ROOT}" \
        ${PACKED_FLAG} \
        --out-dir "${model_dir}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --patch-size "${PATCH_SIZE}" \
        --in-len "${IN_LEN}" \
        --out-len "${OUT_LEN}" \
        --lr "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --alignment "${ALIGNMENT}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --max-steps "${MAX_STEPS}" \
        2>&1 | tee "${train_log}"; then
    status="train-failed"
  fi

  # The newest run directory holds the checkpoint of this invocation.
  run_dir="$(ls -1td "${model_dir}"/*/ 2>/dev/null | head -n 1 || true)"
  checkpoint="${run_dir}last.pt"

  if [ "${status}" = "ok" ] && [ -f "${checkpoint}" ]; then
    if ! "${PYTHON}" scripts/evaluate.py \
          --model "${model}" \
          --checkpoint "${checkpoint}" \
          --data-root "${DATA_ROOT}" \
          ${PACKED_FLAG} \
          --batch-size "${EVAL_BATCH_SIZE}" \
          --num-workers "${EVAL_NUM_WORKERS}" \
          --patch-size "${PATCH_SIZE}" \
          --in-len "${IN_LEN}" \
          --out-len "${OUT_LEN}" \
          --alignment "${ALIGNMENT}" \
          --seed "${SEED}" \
          --device "${DEVICE}" \
          --max-batches "${MAX_BATCHES}" \
          --out-json "${model_dir}/eval_report.json" \
          2>&1 | tee "${eval_log}"; then
      status="eval-failed"
    fi
  elif [ "${status}" = "ok" ]; then
    status="no-checkpoint"
    printf 'no checkpoint found under %s\n' "${model_dir}" | tee "${eval_log}"
  fi

  if [ -f "${model_dir}/eval_report.json" ]; then
    metrics="$(read_report "${model_dir}/eval_report.json" || true)"
  elif [ -f "${eval_log}" ]; then
    metrics="$(read_log_table "${eval_log}" || true)"
  else
    metrics=""
  fi
  if [ -z "${metrics}" ]; then
    metrics="n/a n/a n/a n/a n/a"
  fi

  write_metrics "${model_dir}" "${status}" "${metrics}"

  # shellcheck disable=SC2086
  set -- ${metrics}
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${model}" "${status}" "${1:-n/a}" "${2:-n/a}" "${3:-n/a}" "${4:-n/a}" \
    "${5:-n/a}" >> "${SUMMARY_FILE}"
  set --
done

# ------------------------------------------------------------------ summary --

printf '\n'
printf '================================================================\n'
printf '  SUMMARY\n'
printf '================================================================\n'
printf '%-14s %-14s %9s %9s %9s %9s %9s\n' \
  "MODEL" "STATUS" "CSI@0.5" "CSI@2.0" "ETS@0.5" "FSS@0.5" "DRY@0.5"
printf '%-14s %-14s %9s %9s %9s %9s %9s\n' \
  "--------------" "--------------" "---------" "---------" "---------" \
  "---------" "---------"

while IFS="$(printf '\t')" read -r model status csi05 csi20 ets05 fss05 pers05; do
  printf '%-14s %-14s %9s %9s %9s %9s %9s\n' \
    "${model}" "${status}" "${csi05}" "${csi20}" "${ets05}" "${fss05}" "${pers05}"
done < "${SUMMARY_FILE}"

printf '\nreports : %s/<model>/eval_report.json\n' "${RUNS_DIR}"
printf 'table   : %s\n' "${SUMMARY_FILE}"
