"""Download the score-aware matrix reports and the standard-loss baselines."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import connect, capture  # noqa: E402

MODELS = [
    "simvp_v2", "simvp_v2_nwp_concat", "simvp_v2_quantum", "simvp_v2_nwp_xattn",
    "simvp_v2_nwp_delta", "simvp_early_concat", "simvp_radar_only", "convlstm",
    "predrnn", "earthformer", "metnet3", "diffcast", "cascast_nwp", "mocast",
    "pw_foucast", "fusioncast", "mmf_rnn", "furecast",
]

client = connect()
try:
    for m in MODELS:
        for d in ("score_matrix", "tgrs", "sweep"):
            out = capture(client, f"cat /home/wf/radarNWP/runs/{d}/{m}.eval.json", timeout=120)
            if out.strip().startswith("{"):
                p = Path(__file__).parent / "_evals" / d
                p.mkdir(parents=True, exist_ok=True)
                (p / f"{m}.eval.json").write_text(out, encoding="utf-8")
                print(f"ok {d}/{m}")
            else:
                print(f"MISS {d}/{m}")
    for extra in ("pysteps_adv",):
        for d in ("tgrs", "sweep"):
            out = capture(client, f"cat /home/wf/radarNWP/runs/{d}/{extra}.eval.json", timeout=120)
            if out.strip().startswith("{"):
                p = Path(__file__).parent / "_evals" / d
                p.mkdir(parents=True, exist_ok=True)
                (p / f"{extra}.eval.json").write_text(out, encoding="utf-8")
                print(f"ok {d}/{extra}")
            else:
                print(f"MISS {d}/{extra}")
    # hqma_scan4 D1/D2 for the record
    for d in ("hqma", "hqma_scan4"):
        for name in ("scan_D1", "scan_D2", "scan_D3", "scan_D4", "simvp_v2", "simvp_v2_nwp_concat",
                     "hqma_balanced", "hqma_standard"):
            out = capture(client, f"cat /home/wf/radarNWP/runs/{d}/{name}.eval.json", timeout=120)
            if out.strip().startswith("{"):
                p = Path(__file__).parent / "_evals" / d
                p.mkdir(parents=True, exist_ok=True)
                (p / f"{name}.eval.json").write_text(out, encoding="utf-8")
                print(f"ok {d}/{name}")
finally:
    client.close()
