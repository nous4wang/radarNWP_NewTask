#!/bin/bash
# scripts/run_qlfn2_f38.sh -- Stage F37 s10/s11 curves and submit the 8 new F38 jobs.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
PY="/home/wf/.conda/envs/radar_nwp/bin/python"
SBATCH_FILE="scripts/sbatch_qlfn2_f38.sbatch"
# 1. Strip any accidental CRLF before touching Slurm (pitfall 21)
sed -i 's/\r$//' "${SBATCH_FILE}" "scripts/run_qlfn2_f38.sh"
# 2. Check Slurm controller health first
if ! scontrol ping >/dev/null 2>&1; then
echo "FATAL: scontrol ping failed; controller is not responding" >&2
exit 1
fi
# 3. Run CPU self-checks and judge self-test before submitting
"${PY}" -m qlfn2.checks || exit 1
"${PY}" tools/judge_f38_nwp.py --self-test || exit 1
mkdir -p run_logs/slurm runs/f38_nwp
# 4. Stage completed F37 s10/s11 curves for src_only and evol_both (safe copy)
for arm in src_only evol_both; do
for seed in 10 11; do
src_curve="runs/f37_nwp/${arm}/seed${seed}/curve"
dst_dir="runs/f38_nwp/${arm}/seed${seed}"
if [[ -f "${src_curve}/qlfn2_epoch_curve.json" ]] && [[ ! -f "${dst_dir}/curve/qlfn2_epoch_curve.json" ]]; then
mkdir -p "${dst_dir}/curve"
cp -f "${src_curve}"/*.json "${dst_dir}/curve/"
echo "staged F37 curve: ${src_curve} -> ${dst_dir}/curve/"
fi
done
done
# 5. Submit the 8 new jobs (4 no_nwp_matched + 2 src_only s12/13 + 2 evol_both s12/13)
JOBS=(
"no_nwp_matched:10"
"no_nwp_matched:11"
"no_nwp_matched:12"
"src_only:12"
"no_nwp_matched:13"
"src_only:13"
"evol_both:12"
"evol_both:13"
)
ok_count=0
fail_count=0
for item in "${JOBS[@]}"; do
arm="${item%%:*}"
seed="${item##*:}"
if [[ -f "runs/f38_nwp/${arm}/seed${seed}/curve/qlfn2_epoch_curve.json" ]]; then
echo "skip (already scored): arm=${arm} seed=${seed}"
continue
fi
if jid=$(sbatch --parsable \
--job-name="qlfn2-f38-${arm}-s${seed}" \
--export="ALL,ARM=${arm},SEED=${seed},RUN_ROOT=${REPO_ROOT}/runs/f38_nwp" \
"${SBATCH_FILE}"); then
echo "submitted arm=${arm} seed=${seed} -> job ${jid}"
ok_count=$((ok_count + 1))
else
echo "FAILED to submit arm=${arm} seed=${seed}" >&2
fail_count=$((fail_count + 1))
fi
done
echo "submission summary: ok=${ok_count} failed=${fail_count}"
squeue -u "$USER"
[[ "${fail_count}" -eq 0 ]]
