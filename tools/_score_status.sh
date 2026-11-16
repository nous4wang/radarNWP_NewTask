cd /home/wf/radarNWP
PY=/home/wf/.conda/envs/radar_nwp/bin/python
echo '=== the OLD scan_D1 eval log in full ==='
cat runs/hqma_scan4/scan_D1.eval.log
echo
echo '=== the OLD scan_D2 eval log in full ==='
cat runs/hqma_scan4/scan_D2.eval.log
echo
echo '=== run_hqma_scan4.sh: the eval invocation and its parameters ==='
sed -n '88,125p' scripts/run_hqma_scan4.sh
echo
echo '=== run_hqma_scan4.sh: any train/eval epoch or output-dir setting ==='
grep -nE 'EPOCH|OUT_DIR|TIERS|BATCH|WORKERS|PATCH|SLIDE|LOSS|alpha' scripts/run_hqma_scan4.sh | head -30
date '+%H:%M:%S'
