import json
d = json.load(open("runs/e33_freq_control/summary.json"))
print("per-seed equal-bias diagnostics:")
worst = 0.0
fail_total = 0
for s, e in d["per_seed"].items():
    worst = max(worst, e["equal_bias_max_abs_bias_error"])
    fail_total += e["equal_bias_pairing_failures"]
    print("  seed %s: max_abs_bias_error %.6f  pairing_failures %d/%d (%.2f%%)" % (
        s, e["equal_bias_max_abs_bias_error"], e["equal_bias_pairing_failures"],
        e["n_windows"], 100 * e["equal_bias_pairing_failure_fraction"]))
print("worst over seeds: %.6f" % worst)
print("total pairing failures: %d" % fail_total)
print()
print("bias2 pairs for truthswap_eqbias (arm, control) -- max |diff| per seed:")
cc = d["cross_seed_consistency"]["truthswap_eqbias"]
for i, (a, b) in enumerate(cc["bias2_pairs"]):
    print("  seed %d: %.6f vs %.6f  |diff| %.2e" % (i, a, b, abs(a - b)))
