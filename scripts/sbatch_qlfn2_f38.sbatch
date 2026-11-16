#!/bin/bash
#SBATCH --job-name=qlfn2-f38
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=run_logs/slurm/qlfn2-f38-%j.out
#SBATCH --error=run_logs/slurm/qlfn2-f38-%j.err

# F38 -- architecture-matched zero-NWP control (no_nwp_matched) + 4 seeds
# (10, 11, 12, 13) to quantify the inter-seed variance band.
#
# Cloned verbatim from sbatch_qlfn2_f37.sbatch so that every training and
# scoring flag (--packed, --num-workers 4, 64/64 geometry, thresholds,
# fss_radii) is 100 % identical to F37.
#
# Three arms x 4 seeds = 12 runs (4 reused from F37 s10/s11 + 8 new jobs):
#   src_only        vel=false src=true   (full NWP)
#   evol_both       vel=true  src=true   (full NWP)
#   no_nwp_matched  vel=false src=true + --no-nwp (architecture-matched control)

# conda's activation scripts reference unset variables, so set -u must be off
# during activation (MKL_INTERFACE_LAYER: unbound variable otherwise).
set -eo pipefail

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate radar_nwp
set -u

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
BENCH_ROOT="${BENCH_ROOT:-/home/wf/radarNWP}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/f38_nwp}"

ARM="${ARM:?ARM must be set (src_only|evol_both|no_nwp_matched)}"
ARM_NAME="${ARM}"
SEED="${SEED:-10}"
EPOCHS="${EPOCHS:-10}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-4}"
QPE_WEIGHT="${QPE_WEIGHT:-0.5}"
THRESHOLDS="${THRESHOLDS:-0.1 0.5 1 2 4 8 16}"
FSS_RADII="${FSS_RADII:-1 2 4}"

case "${ARM}" in
    src_only)
        ARM_KWARGS='{"use_halo":false,"use_nwp_gate":false,"use_nwp_vel":false,"use_nwp_source":true}'
        TRAIN_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        SCORE_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        ;;
    evol_both)
        ARM_KWARGS='{"use_halo":false,"use_nwp_gate":false,"use_nwp_vel":true,"use_nwp_source":true}'
        TRAIN_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        SCORE_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        ;;
    no_nwp_matched)
        ARM_KWARGS='{"use_halo":false,"use_nwp_gate":false,"use_nwp_vel":false,"use_nwp_source":true}'
        TRAIN_EXTRA=(--model-kwargs "${ARM_KWARGS}" --no-nwp)
        SCORE_EXTRA=(--model-kwargs "${ARM_KWARGS}" --no-nwp)
        ;;
    *)
        echo "unknown arm ${ARM}" >&2
        exit 2
        ;;
esac
export ARM_NAME ARM_KWARGS

OUT_DIR="${RUN_ROOT}/${ARM_NAME}/seed${SEED}"
mkdir -p "${OUT_DIR}" run_logs/slurm

echo "=========================================================="
echo " F38 architecture-matched 4-seed  $(date -Is)"
echo "   job id      : ${SLURM_JOB_ID}"
echo "   arm         : ${ARM_NAME}"
echo "   model-kwargs: ${ARM_KWARGS}"
echo "   train flags : ${TRAIN_EXTRA[*]:-<none>}"
echo "   eval flags  : ${SCORE_EXTRA[*]:-<none>}"
echo "   seed        : ${SEED}   epochs: ${EPOCHS}"
echo "   geometry    : patch ${PATCH_SIZE} / stride ${SLIDE_STRIDE}"
echo "   out         : ${OUT_DIR}"
echo "   python      : $(command -v python)"
echo "   cuda visible: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================================="

echo
echo "--- arm identity + architecture-match preflight (before GPU time) ----"
python - <<'PY'
import json
import os
import sys
import torch

from qlfn2.qlfn2_model import (
    HALO_CONTEXT_INDICES, NWP_VARS, NWP_EVOL_INDICES, QLFN2Adapter,
    archive_to_net,
)

assert HALO_CONTEXT_INDICES == (2, 3, 0, 14, 23), "channel table drifted"
assert NWP_VARS[22] == "TTdMax74" and NWP_VARS[23] == "TTdMean74"
assert len(NWP_EVOL_INDICES) == 10
print("NWP_VARS[22:24]      :", NWP_VARS[22:24])
print("HALO_CONTEXT_INDICES :", HALO_CONTEXT_INDICES)
print("NWP_EVOL_INDICES     :", NWP_EVOL_INDICES)

name = os.environ["ARM_NAME"]
kwargs = json.loads(os.environ.get("ARM_KWARGS", "{}"))
torch.manual_seed(103)
model = QLFN2Adapter(**kwargs).eval()
c = model.core
state = (c.halo_proj is not None, c.nwp_gate is not None,
         c.nwp_vel_net is not None, c.nwp_source_net is not None)
want = {
    "evol_both":      (False, False, True, True),
    "src_only":       (False, False, False, True),
    "no_nwp_matched": (False, False, False, True),
}[name]
n_params = sum(p.numel() for p in model.parameters())
want_params = 88237 if name == "evol_both" else 85019
print(f"arm {name}: halo/gate/vel/src={state} want={want} "
      f"state_dict_keys={len(model.state_dict())} params={n_params}")
if state != want or n_params != want_params:
    print(f"FATAL: arm construction mismatch (state={state}, params={n_params})")
    sys.exit(4)

# Discriminating gate for F38: no_nwp_matched and src_only must have the
# EXACT same state_dict key set (30 keys) and parameter count (85,019),
# whereas F37's default no_nwp had halo_proj + nwp_gate (32 keys, 83,251).
src_ref = QLFN2Adapter(
    use_halo=False, use_nwp_gate=False, use_nwp_vel=False, use_nwp_source=True
).eval()
if name in ("src_only", "no_nwp_matched"):
    if set(model.state_dict().keys()) != set(src_ref.state_dict().keys()):
        print("FATAL: state_dict keys do not match src_only")
        sys.exit(4)

# Verify lead-0 guard and zero-NWP identity on non-trivial weights
with torch.no_grad():
    model.core.nwp_source_net[-1].weight.normal_(0.0, 0.1)
    model.core.nwp_source_net[-1].bias.normal_(0.0, 0.1)
raw = torch.zeros(2, 11, 10, 16, 16)
raw[:, :, 6] = 300.0
raw[:, :, 9] = 200.0
comp = archive_to_net(raw)
nwp = torch.rand(2, 2, 27, 16, 16)
with torch.no_grad():
    p_nwp = model(comp, nwp)
    p_zero = model(comp, torch.zeros_like(nwp))
    p_none = model(comp, None)
gap_l0 = float((p_nwp[:, 0] - p_zero[:, 0]).abs().max())
gap_zero_none = float((p_zero - p_none).abs().max())
gap_late = float((p_nwp[:, 1:] - p_zero[:, 1:]).abs().max())
print(f"lead-0 gap={gap_l0:.3e}  zero_vs_none={gap_zero_none:.3e}  late_active_gap={gap_late:.3e}")
if gap_l0 != 0.0 or gap_zero_none != 0.0 or gap_late <= 0.0:
    print("FATAL: lead-0 guard or zero-NWP ablation identity failed")
    sys.exit(4)
print("diagnostics           :", model.last_diagnostics)
PY
preflight_rc=$?
echo "preflight rc=${preflight_rc}"
if [[ "${preflight_rc}" -ne 0 ]]; then
    exit 4
fi

echo
echo "--- self-checks (31/31 required) -------------------------------------"
python -m qlfn2.checks
checks_rc=$?
echo "checks rc=${checks_rc}"
if [[ "${checks_rc}" -ne 0 ]]; then
    exit 5
fi

python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         :", torch.cuda.get_device_name(0))
PY

echo
echo "--- train (every epoch kept) -----------------------------------------"
python -u "${REPO_ROOT}/scripts/train_qlfn2_keep.py" \
    --model qlfn2 \
    --data-root "${DATA_ROOT}" \
    --packed \
    --split train \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${WORKERS}" \
    --patch-size "${PATCH_SIZE}" \
    --slide-stride "${SLIDE_STRIDE}" \
    --lr 2e-4 \
    --loss balanced \
    --qpe-weight "${QPE_WEIGHT}" \
    --log-every 100 \
    --seed "${SEED}" \
    --device cuda \
    --out-dir "${OUT_DIR}/train" \
    ${TRAIN_EXTRA[@]+"${TRAIN_EXTRA[@]}"}
train_rc=$?
echo "train rc=${train_rc}"

echo
echo "--- locate the checkpoints -------------------------------------------"
CKPT_DIR="$(ls -1dt "${OUT_DIR}"/train/*/ 2>/dev/null | head -1)"
if [[ -z "${CKPT_DIR}" ]]; then
    echo "FATAL: no checkpoint directory under ${OUT_DIR}/train" >&2
    exit 3
fi
echo "checkpoint dir: ${CKPT_DIR}"
ls -1 "${CKPT_DIR}" | sed 's/^/   /'

echo
echo "--- score every epoch (same flags as training) ------------------------"
python -u "${REPO_ROOT}/tools/score_epoch_curve.py" \
    --model qlfn2 \
    --checkpoint-dir "${CKPT_DIR}" \
    --data-root "${DATA_ROOT}" \
    --out-dir "${OUT_DIR}/curve" \
    --split test \
    --patch-size "${PATCH_SIZE}" \
    --slide-stride "${SLIDE_STRIDE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${WORKERS}" \
    --thresholds ${THRESHOLDS} \
    --fss-radii ${FSS_RADII} \
    --seed "${SEED}" \
    --device cuda \
    ${SCORE_EXTRA[@]+"${SCORE_EXTRA[@]}"}
score_rc=$?
echo "score rc=${score_rc}"

echo
echo "--- report check (per-lead data must be present for G0/G1) -----------"
CURVE="${OUT_DIR}/curve/qlfn2_epoch_curve.json"
python - "${CURVE}" "${OUT_DIR}/curve" <<'PY'
import json
import sys
from pathlib import Path

summary_path, curve_dir = Path(sys.argv[1]), Path(sys.argv[2])
if not summary_path.is_file():
    print(f"FATAL: no curve summary at {summary_path}")
    raise SystemExit(5)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
rows = [r for r in summary["rows"] if r.get("epoch") is not None
        and "csi_2.0" in r]
reports = sorted(curve_dir.glob("qlfn2_epoch_*.json"))
reports = [p for p in reports if "epoch_curve" not in p.name and "last" not in p.name]
print(f"geometry       : {summary['geometry']}")
print(f"scored epochs  : {len(rows)}")
print(f"n_samples      : {sorted({r.get('n_samples') for r in rows})}")
print(f"per-lead files : {len(reports)}")
if not rows or len(rows) != 10:
    print("FATAL: expected 10 scored epoch checkpoints")
    raise SystemExit(6)
if summary["geometry"] != {"patch_size": 64, "slide_stride": 64}:
    print("FATAL: geometry is not 64/64")
    raise SystemExit(7)
if {r.get("n_samples") for r in rows} != {538}:
    print("FATAL: n_samples is not 538")
    raise SystemExit(8)
if len(reports) != 10:
    print("FATAL: G0/G1 need 10 per-lead epoch reports")
    raise SystemExit(9)
for p in reports:
    d = json.loads(p.read_text(encoding="utf-8"))
    lc = d.get("lead_csi")
    if not isinstance(lc, dict) or len(lc) != 10:
        print(f"FATAL: {p.name} has no 10-lead lead_csi")
        raise SystemExit(10)
    for k in range(10):
        v = lc.get(str(k), {}).get("2.0")
        if not isinstance(v, (int, float)):
            print(f"FATAL: {p.name} lead {k} has no CSI@2.0")
            raise SystemExit(10)
print(f"peak           : {summary['peak_checkpoint']}  "
      f"CSI@2.0 = {summary['peak_csi2']:.4f}")
PY
report_rc=$?
echo "report rc=${report_rc}"

echo
echo "=========================================================="
echo " arm ${ARM_NAME} seed ${SEED} finished $(date -Is)"
echo "   train rc = ${train_rc}   score rc = ${score_rc}   report rc = ${report_rc}"
echo "   curve    = ${CURVE}"
echo "=========================================================="

exit $(( train_rc | score_rc | report_rc ))
