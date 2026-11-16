FLAG='--topo-clamp-mm 20 --model-kwargs {"clamp_mm":20.0}'
read -r -a A <<< "$FLAG"
printf 'N=%s\n' "${#A[@]}"
for i in "${!A[@]}"; do printf '[%s]=%s\n' "$i" "${A[$i]}"; done
