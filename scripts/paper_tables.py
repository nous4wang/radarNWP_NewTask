"""Build the paper's tables from the evaluation reports.

The reports are JSON files written by ``scripts/evaluate.py``. This script
turns them into the three tables the paper needs:

  * the main comparison, one row per model, with the scores at several
    thresholds and the CSI at each lead time;
  * the robustness comparison, the same models on the cases whose radar
    series has a hole in it;
  * the ablation ladder, the models that differ only in how the NWP enters.

It also prints the delta between a plugin-mounted model and its plain
backbone, because that delta is the claim.

Usage::

    python scripts/paper_tables.py --runs-dir runs/tgrs
    python scripts/paper_tables.py --runs-dir runs/tgrs --markdown paper.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: The four tiers of the main comparison, in the order the paper presents
#: them. A model that is not in the table is listed separately.
TIERS: list[tuple[str, list[str]]] = [
    ("Physics extrapolation", ["pysteps_adv"]),
    ("Classic spatio-temporal", [
        "convlstm", "predrnn", "simvp_v2", "earthformer",
    ]),
    ("Multimodal SOTA", [
        "metnet3", "diffcast", "cascast_nwp", "mocast",
        "pw_foucast", "fusioncast", "mmf_rnn", "furecast",
    ]),
    ("Q-NWP coupling", [
        "simvp_v2_quantum", "simvp_v2_nwp_concat",
        "simvp_v2_nwp_xattn", "simvp_v2_nwp_delta",
    ]),
]

#: The ablation ladder: same backbone, different ways of using the NWP.
#: M0 is the radar-only twin of the SimVP-v2 core -- the same network with
#: the fields taken away -- so it says how much of the gap is the network
#: and how much of it is the NWP. The registered name is
#: ``simvp_radar_only``; writing ``simvp_v2_radar_only`` here left the row
#: permanently empty because no such model has ever been trained.
ABLATION: list[tuple[str, str]] = [
    ("M0  radar only", "simvp_radar_only"),
    ("M1  naive concat", "simvp_v2_nwp_concat"),
    ("M2  cross-attention", "simvp_v2_nwp_xattn"),
    ("M3  tendency difference", "simvp_v2_nwp_delta"),
    ("M4  full Q-NWP plugin", "simvp_v2_quantum"),
]


def load_report(runs_dir: Path, model: str, suffix: str) -> dict | None:
    path = runs_dir / f"{model}{suffix}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fmt(value: float, width: int = 6) -> str:
    if value is None:
        return "n/a".rjust(width)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a".rjust(width)
    if v != v:  # NaN
        return "n/a".rjust(width)
    return f"{v:.3f}".rjust(width)


def score_at(report: dict, threshold: str, key: str) -> float:
    return report.get("thresholds", {}).get(threshold, {}).get(key, float("nan"))


def lead_csi(report: dict, threshold: str) -> list[float]:
    scores = report.get("lead_scores") or {}
    out = []
    for lead in sorted(scores, key=lambda k: int(k)):
        out.append(scores[lead].get(threshold, {}).get("csi", float("nan")))
    return out


def lead_far(report: dict, threshold: str) -> list[float]:
    scores = report.get("lead_scores") or {}
    out = []
    for lead in sorted(scores, key=lambda k: int(k)):
        out.append(scores[lead].get(threshold, {}).get("far", float("nan")))
    return out


def reference_table(runs_dir: Path, thresholds: list[str]) -> str:
    """The floor the models have to clear.

    Two references sit below the trained models, and both belong in the
    same table as the models: without them the reader sees a score of
    0.004 and concludes the method is very bad, when the honest
    comparison is against 0.0924.

    The persistence forecast holds the last observed reflectivity still
    for every lead time and converts it with the same Z-R relation the
    analytical baseline uses. It is not a strawman -- it is what a
    forecaster gets by doing nothing clever, and on this benchmark it
    beats every deep model at the heavy thresholds.

    The analytical extrapolation is the physical baseline the matrix was
    built around.
    """
    lines: list[str] = []
    header = (
        f"{'reference':24s}" + "".join(
            f"{'CSI@' + t:>9s}{'FAR@' + t:>9s}" for t in thresholds
        )
    )
    lines.append(header)
    lines.append("-" * len(header))

    for name in ("persistence", "pysteps_adv"):
        report = load_report(runs_dir, name, ".eval")
        if report is None:
            lines.append(f"  {name:22s} (no report)")
            continue
        row = f"  {name:22s}"
        for thr in thresholds:
            row += fmt(score_at(report, thr, "csi"), 9)
            row += fmt(score_at(report, thr, "far"), 9)
        lines.append(row)

    lines.append("")
    lines.append(
        "The persistence reference is what a forecaster gets by holding the"
    )
    lines.append(
        "last observed reflectivity still. Anything below it has not learned"
    )
    lines.append(
        "to place the heavy rain; it has only learned to under-predict."
    )
    return "\n".join(lines)


def main_table(runs_dir: Path, thresholds: list[str]) -> str:
    lines: list[str] = []
    header = (
        f"{'model':24s}" + "".join(
            f"{'CSI@' + t:>9s}{'FAR@' + t:>9s}" for t in thresholds
        )
    )
    lines.append(header)
    lines.append("-" * len(header))

    found: list[str] = []
    for tier, models in TIERS:
        lines.append(f"{tier}")
        for model in models:
            report = load_report(runs_dir, model, ".eval")
            if report is None:
                lines.append(f"  {model:22s} (no report)")
                continue
            found.append(model)
            row = f"  {model:22s}"
            for thr in thresholds:
                row += fmt(score_at(report, thr, "csi"), 9)
                row += fmt(score_at(report, thr, "far"), 9)
            lines.append(row)
        lines.append("")

    # Anything that ran but is not in the tiers.
    others = []
    for path in sorted(runs_dir.glob("*.eval.json")):
        model = path.name.replace(".eval.json", "")
        if model not in found:
            others.append(model)
    if others:
        lines.append("not in the tiers")
        for model in others:
            report = load_report(runs_dir, model, ".eval")
            if report is None:
                continue
            row = f"  {model:22s}"
            for thr in thresholds:
                row += fmt(score_at(report, thr, "csi"), 9)
                row += fmt(score_at(report, thr, "far"), 9)
            lines.append(row)
    return "\n".join(lines)


def lead_table(runs_dir: Path, threshold: str, models: list[str]) -> str:
    """CSI and FAR per lead time -- the two axes the paper's claim runs on."""
    lines: list[str] = []
    leads = sorted(
        set(
            lead
            for m in models
            for lead in (load_report(runs_dir, m, ".eval") or {}).get(
                "lead_scores", {}
            )
        ),
        key=lambda k: int(k),
    )
    header = f"{'model':24s}" + "".join(
        f"{'T+' + str(int(l) * 6) + 'm':>9s}" for l in leads
    )
    lines.append(f"CSI at {threshold} mm")
    lines.append(header)
    lines.append("-" * len(header))
    for model in models:
        report = load_report(runs_dir, model, ".eval")
        if report is None:
            continue
        values = lead_csi(report, threshold)
        row = f"  {model:22s}"
        for v in values:
            row += fmt(v, 9)
        lines.append(row)

    lines.append("")
    lines.append(f"FAR at {threshold} mm")
    lines.append(header)
    lines.append("-" * len(header))
    for model in models:
        report = load_report(runs_dir, model, ".eval")
        if report is None:
            continue
        values = lead_far(report, threshold)
        row = f"  {model:22s}"
        for v in values:
            row += fmt(v, 9)
        lines.append(row)
    return "\n".join(lines)


def ablation_table(runs_dir: Path, thresholds: list[str]) -> str:
    lines: list[str] = []
    header = (
        f"{'variant':30s}" + "".join(f"{'CSI@' + t:>9s}" for t in thresholds)
        + f"{'n':>7s}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for label, model in ABLATION:
        report = load_report(runs_dir, model, ".eval")
        if report is None:
            lines.append(f"  {label:28s} (no report)")
            continue
        row = f"  {label:28s}"
        for thr in thresholds:
            row += fmt(score_at(report, thr, "csi"), 9)
        row += f"{report.get('n_samples', 0):7d}"
        lines.append(row)
    return "\n".join(lines)


def robustness_table(runs_dir: Path, thresholds: list[str]) -> str:
    lines: list[str] = []
    header = (
        f"{'model':24s}" + "".join(
            f"{'clean CSI@' + t:>13s}{'robust CSI@' + t:>13s}"
            for t in thresholds
        )
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, models in TIERS:
        for model in models:
            clean = load_report(runs_dir, model, ".eval")
            robust = load_report(runs_dir, model, ".robustness")
            if clean is None and robust is None:
                continue
            row = f"  {model:22s}"
            for thr in thresholds:
                row += fmt(
                    score_at(clean, thr, "csi") if clean else float("nan"), 13
                )
                row += fmt(
                    score_at(robust, thr, "csi") if robust else float("nan"), 13
                )
            lines.append(row)
    return "\n".join(lines)


def delta_table(runs_dir: Path, thresholds: list[str]) -> str:
    """The plugin against its own backbone -- a controlled comparison."""
    pairs = [
        ("simvp_v2_quantum", "simvp_v2"),
        ("simvp_v2_nwp_concat", "simvp_v2"),
        ("simvp_v2_nwp_xattn", "simvp_v2"),
        ("simvp_v2_nwp_delta", "simvp_v2"),
    ]
    lines: list[str] = []
    header = (
        f"{'with NWP':24s}{'baseline':>14s}"
        + "".join(f"{'dCSI@' + t:>10s}" for t in thresholds)
        + f"{'dFAR@0.5':>10s}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for with_nwp, baseline in pairs:
        a = load_report(runs_dir, with_nwp, ".eval")
        b = load_report(runs_dir, baseline, ".eval")
        if a is None or b is None:
            lines.append(f"  {with_nwp:22s} (incomplete)")
            continue
        row = f"  {with_nwp:22s}{baseline:>14s}"
        for thr in thresholds:
            d = score_at(a, thr, "csi") - score_at(b, thr, "csi")
            row += f"{d:+10.3f}" if d == d else "     n/a"
        dfar = score_at(a, "0.5", "far") - score_at(b, "0.5", "far")
        row += f"{dfar:+10.3f}" if dfar == dfar else "     n/a"
        lines.append(row)
    return "\n".join(lines)


def underprediction_table(runs_dir: Path, threshold: str = "0.5") -> str:
    """How much rain does each model actually predict?

    The verification scores answer "how well did the model do", but they
    do not say *what* the model did. A model that predicts almost no rain
    scores a poor CSI, and a model that predicts rain in the wrong places
    scores the same poor CSI -- and the two have different remedies.

    The frequency bias separates them: it is the ratio of the rain the
    model predicted to the rain that fell. A value near one means the
    total is right, and a low value means the model is systematically
    leaving rain out.
    """
    lines: list[str] = []
    header = (
        f"{'model':24s}{'CSI':>8s}{'POD':>8s}{'FAR':>8s}{'BIAS':>8s}"
        f"{'hits':>10s}{'misses':>10s}"
    )
    lines.append(f"under-prediction at {threshold} mm")
    lines.append(header)
    lines.append("-" * len(header))
    rows = []
    for path in sorted(runs_dir.glob("*.eval.json")):
        model = path.name.replace(".eval.json", "")
        report = load_report(runs_dir, model, ".eval")
        if report is None:
            continue
        entry = report.get("thresholds", {}).get(threshold, {})
        rows.append((
            model,
            entry.get("csi", float("nan")),
            entry.get("pod", float("nan")),
            entry.get("far", float("nan")),
            entry.get("bias", float("nan")),
            entry.get("hits", 0),
            entry.get("misses", 0),
        ))
    rows.sort(key=lambda r: (-r[4] if r[4] == r[4] else 1e9))
    for model, csi, pod, far, bias, hits, misses in rows:
        lines.append(
            f"  {model:22s}{fmt(csi, 8)}{fmt(pod, 8)}{fmt(far, 8)}"
            f"{fmt(bias, 8)}{int(hits):10d}{int(misses):10d}"
        )
    lines.append("")
    lines.append("  BIAS = (hits + false alarms) / (hits + misses);")
    lines.append("  below 1 means the model predicts less rain than fell.")
    return "\n".join(lines)


def rapsd_table(runs_dir: Path, rapsd_dir: Path) -> str:
    """The spectral diagnostic, next to the scores it explains.

    The verification scores say how well a model did. The spectrum says
    what it did to the field. Putting the two side by side is the point:
    a model that keeps the small-scale energy is a model that can predict
    the convective core, and one that does not is predicting a blurred
    version of the storm.

    A high_k ratio below one means the forecast lost the small-scale
    energy; above one means it added some, which is a noise floor rather
    than structure.
    """
    rows = []
    for path in sorted(rapsd_dir.glob("*.json")):
        model = path.stem
        report = load_report(runs_dir, model, ".eval")
        try:
            rapsd = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        summary = rapsd.get("summary", {})
        high = summary.get("high_k_ratio", float("nan"))
        low = summary.get("low_k_ratio", float("nan"))
        csi = (
            score_at(report, "2.0", "csi") if report else float("nan")
        )
        rows.append((model, high, low, csi))

    if not rows:
        return "(no spectral reports found)"

    rows.sort(key=lambda r: r[1] if r[1] == r[1] else 1e9)
    lines = [
        f"{'model':26s}{'high_k (<10km)':>17s}{'low_k (>30km)':>15s}"
        f"{'CSI@2.0':>10s}",
        "-" * 68,
    ]
    for model, high, low, csi in rows:
        lines.append(
            f"  {model:24s}{fmt(high, 17)}{fmt(low, 15)}{fmt(csi, 10)}"
        )

    # The correlation between the two, over the models that have both.
    pairs = [
        (high, csi) for _, high, _, csi in rows
        if high == high and csi == csi
    ]
    if len(pairs) >= 3:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in pairs)
        den = (
            (sum((x - mx) ** 2 for x in xs) ** 0.5)
            * (sum((y - my) ** 2 for y in ys) ** 0.5)
        )
        r = num / den if den > 0 else float("nan")
        lines.append("")
        lines.append(
            f"  corr(high_k ratio, CSI@2.0) over {len(pairs)} models = {r:+.3f}"
        )
        lines.append(
            "  (a positive correlation means the models that keep the small-scale"
        )
        lines.append(
            "   energy are the ones that score at the heavy thresholds)"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument(
        "--thresholds", nargs="*", default=["0.1", "0.5", "2.0", "4.0"]
    )
    parser.add_argument("--markdown", default=None)
    parser.add_argument(
        "--rapsd-dir", default=None,
        help="a directory of rapsd.json reports, one per model",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"ERROR: {runs_dir} is not a directory")
        return 2

    sections: list[tuple[str, str]] = [
        ("The floor: what doing nothing clever scores",
         reference_table(runs_dir, args.thresholds)),
        ("Main comparison (clean test split, tiles stitched to 301x301)",
         main_table(runs_dir, args.thresholds)),
        ("Under-prediction: how much rain each model actually predicts",
         underprediction_table(runs_dir, "0.5")),
        ("Lead-time breakdown", lead_table(
            runs_dir, "0.5", [m for _, models in TIERS for m in models]
        )),
        ("Ablation ladder", ablation_table(runs_dir, args.thresholds)),
        ("Robustness (cases with a hole in the radar series)",
         robustness_table(runs_dir, args.thresholds)),
        ("The controlled comparison", delta_table(runs_dir, args.thresholds)),
    ]

    if args.rapsd_dir:
        sections.append((
            "Spectral diagnostic: what the models did to the field",
            rapsd_table(runs_dir, Path(args.rapsd_dir)),
        ))

    text = "\n\n".join(f"## {title}\n\n{body}" for title, body in sections)
    print(text)

    if args.markdown:
        Path(args.markdown).write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten to {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
