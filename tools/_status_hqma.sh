cd /home/wf/radarNWP
echo '=== 训练 ==='
for r in scan_A1 scan_A2 scan_A3 scan_A4; do
  printf '%-10s ' "$r"
  grep -E '^epoch [0-9]+:' runs/hqma_scan/$r.log 2>/dev/null | tail -1 || echo '(none)'
done
echo '=== eval ==='
ls runs/hqma_scan/*.eval.json 2>/dev/null | wc -l
echo '=== 进程 ==='
ps -eo args | grep -E '[t]rain.py|[e]valuate.py' | grep -oE '\-\-model [a-z_0-9_]+' | cut -d' ' -f2 | sort -u | tr '\n' ' '
echo
echo '=== GPU ==='
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '
echo
echo '=== driver ==='
tail -25 runs/hqma_scan_driver.log 2>/dev/null
echo '=== files ==='
ls -la runs/hqma_scan/ 2>/dev/null | head -60
