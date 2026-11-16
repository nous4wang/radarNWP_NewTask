"""E45 round-34 verification.

What the previous 33 rounds did NOT check
------------------------------------------
They re-read the ARTEFACTS and the DOCUMENTS and confirmed the version
pointers.  Two things stayed unverified:

A. The S3 verdict was only ever read from PROSE.  If a document ever
   over- or under-states the S3 outcome, nothing would catch it.  Here
   the verdict is re-derived from the RAW artefact
   (``tools/_evals/e45/s3_equalbias.json``) and compared against what
   the documents claim.

B. The conclusion SEMANTICS were never compared across the whole
   authoritative chain.  Now every place that states the E45 conclusion
   must carry the negative verdict, the forbidden claims must be absent,
   and the version pointers must be mutually consistent and not older
   than v2.46 (equality with a fixed string would turn a legal bump into
   a false failure).

Plus the three registered implementation defects: "the fix has been
located" and "the fix has been implemented" are DIFFERENT claims, so
the implementation status is reported explicitly.

Read-only.  Writes nothing, submits nothing, touches no remote state.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

DOCS = {
    "MASTER": ROOT / "docs" / "MASTER.md",
    "INDEX": ROOT / "docs" / "experiments" / "INDEX.md",
    "E45DOC": ROOT / "docs" / "experiments" /
    "E45-gate-direction-amount-separation-baseline.md",
    "MEMORY": pathlib.Path(r"R:\1radarNWPprediction\.workbuddy\memory\MEMORY.md"),
    "HANDOVER": pathlib.Path(r"R:\1radarNWPprediction\.workbuddy\memory\HANDOVER_E45.md"),
}

FAILS = []
WARNS = []


def ok(msg):
    print(f"  PASS  {msg}")


def bad(msg):
    FAILS.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg):
    WARNS.append(msg)
    print(f"  WARN  {msg}")


def info(msg):
    print(f"  INFO  {msg}")


# --------------------------------------------------------------------------
# A. S3 verdict re-derived from the raw artefact
# --------------------------------------------------------------------------
def check_s3():
    print("\n=== A. S3 equal-bias verdict re-derived from the raw artefact ===")
    path = ROOT / "tools" / "_evals" / "e45" / "s3_equalbias.json"
    if not path.exists():
        bad(f"s3_equalbias.json missing locally ({path})")
        return
    s3 = json.loads(path.read_text(encoding="utf-8"))

    if s3.get("control") != "B0":
        bad(f"S3 control is {s3.get('control')!r}, expected 'B0'")
    else:
        ok(f"S3 control = B0 (threshold {s3.get('threshold_mm')} mm, "
           f"epoch {s3.get('epoch')}, n_boot {s3.get('n_boot')})")

    arms = s3.get("arms", {})
    if len(arms) != 6:
        bad(f"S3 covers {len(arms)} arms, expected 6 (A1,A2,A3a,A3b,A4,A4b)")
    else:
        ok("S3 covers all 6 arms")

    allowed_verdicts = {"GAIN_WAS_AREA_OR_AMPLITUDE", "NO_GAIN_TO_BEGIN_WITH"}
    qualified = []
    for name in sorted(arms):
        a = arms[name]
        raw, eb = a.get("raw", {}), a.get("equal_bias", {})
        cr, ce = raw.get("crosses_zero"), eb.get("crosses_zero")
        print(f"        {name:4s} raw   point={raw.get('point'):+.6f} "
              f"CI=[{raw.get('ci_lo'):+.6f}, {raw.get('ci_hi'):+.6f}] cross0={cr}")
        print(f"        {name:4s} ebias point={eb.get('point'):+.6f} "
              f"CI=[{eb.get('ci_lo'):+.6f}, {eb.get('ci_hi'):+.6f}] cross0={ce} "
              f"verdict={a.get('s3_verdict')}")
        if raw.get("n_seeds") != 3 or eb.get("n_seeds") != 3:
            bad(f"{name}: S3 pooled over {raw.get('n_seeds')}/{eb.get('n_seeds')} "
                f"seeds, expected 3")
        if not (cr is True and ce is True):
            qualified.append(name)
        if a.get("s3_verdict") not in allowed_verdicts:
            bad(f"{name}: unknown s3_verdict {a.get('s3_verdict')!r}")
        if a.get("s3_verdict") == "GAIN_WAS_AREA_OR_AMPLITUDE" and not (cr and ce):
            bad(f"{name}: verdict claims an area/amplitude effect but a CI "
                f"does not cross zero")

    if qualified:
        bad(f"arms whose S3 CI does NOT cross zero (would qualify for 10 seed): "
            f"{qualified}")
    else:
        ok("all 6 arms: raw AND equal-bias CI cross zero -> "
           "qualified set EMPTY (no arm advances to 10 seed)")

    # the verdict strings themselves
    verdicts = {n: arms[n].get("s3_verdict") for n in sorted(arms)}
    area = [n for n, v in verdicts.items()
            if v == "GAIN_WAS_AREA_OR_AMPLITUDE"]
    nogain = [n for n, v in verdicts.items()
              if v == "NO_GAIN_TO_BEGIN_WITH"]
    info(f"area/amplitude-effect arms ({len(area)}): {area}")
    info(f"no-gain-to-begin-with arms ({len(nogain)}): {nogain}")


# --------------------------------------------------------------------------
# B. conclusion semantics across the authoritative chain
# --------------------------------------------------------------------------
def sections(text):
    out, head, buf = [], None, []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            if head is not None:
                out.append((head, "\n".join(buf)))
            head, buf = line.strip(), []
        else:
            buf.append(line)
    if head is not None:
        out.append((head, "\n".join(buf)))
    return out


def vtuple(text):
    m = re.search(r"v(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def region_for(key, text, secs):
    """Text region that is supposed to carry the E45 conclusion."""
    if key == "MASTER":
        body = []
        for head, sec in secs:
            if "4.47" in head or re.match(r"^#{1,4}\s*9\b", head) \
                    or re.match(r"^#{1,4}\s*12\b", head):
                body.append(sec)
        body.append("\n".join(l for l in text.splitlines()
                               if "E45" in l))
        return "\n".join(body)
    if key == "INDEX":
        return "\n".join(l for l in text.splitlines()
                         if "E45" in l and l.strip().startswith("|"))
    if key == "E45DOC":
        return text
    if key == "MEMORY":
        return "\n".join(l for l in text.splitlines() if "E45" in l)
    if key == "HANDOVER":
        return text


NEGATIVE = re.compile(r"(判负|跨零|无一臂|全部判负)")
FORBIDDEN = [
    "已实现 SOTA", "已解决 novelty", "C0 被推翻",
    "LABEL 有孤立针状尖峰",
]


def check_semantics():
    print("\n=== B. conclusion semantics across the authoritative chain ===")
    texts, secs = {}, {}
    for key, path in DOCS.items():
        if not path.exists():
            bad(f"{key}: file missing ({path})")
            continue
        texts[key] = path.read_text(encoding="utf-8")
        secs[key] = sections(texts[key])

    # version pointers: mutually consistent and not older than v2.46
    # NOTE: the version marker does NOT live in the head of every document.
    # MASTER/MEMORY/HANDOVER carry it in the head; INDEX carries it in the
    # E45 table row; the E45 experiment file carries it inside the body.
    # Searching only the head would be a false failure -- so each document
    # is read at the place where its own marker actually lives.
    print("\n  -- version pointers --")
    vers = {}

    def version_region(key, text):
        if key == "INDEX":
            rows = [l for l in text.splitlines()
                    if l.strip().startswith("|") and "E45" in l]
            return "\n".join(rows)
        if key == "E45DOC":
            return text
        return "\n".join(text.splitlines()[:20])

    def vmax(text):
        found = [tuple(int(x) for x in m)
                 for m in re.findall(r"v(\d+)\.(\d+)", text or "")]
        return max(found) if found else None

    for key, text in texts.items():
        v = vmax(version_region(key, text))
        vers[key] = v
        if v is None:
            bad(f"{key}: no version token in its version region")
        else:
            info(f"{key} version = v{v[0]}.{v[1]}")
    known = [v for v in vers.values() if v]
    if known and any(v != known[0] for v in known):
        bad(f"version pointers disagree: {vers}")
    else:
        ok("all version pointers agree")
    if known and known[0] < (2, 46):
        bad(f"version pointer v{known[0][0]}.{known[0][1]} is older than v2.46")
    elif known:
        ok(f"version pointer not older than v2.46 (v{known[0][0]}.{known[0][1]})")

    # the negative verdict must be stated in every authoritative region
    print("\n  -- negative verdict present in every region --")
    for key, text in texts.items():
        region = region_for(key, text, secs.get(key, []))
        if NEGATIVE.search(region):
            ok(f"{key}: states the negative verdict (判负 / 跨零 / 无一臂)")
        else:
            bad(f"{key}: no negative-verdict statement found in its E45 region")

    # forbidden claims must be ABSENT as ASSERTIONS.
    # The literal phrase may legitimately appear inside a PROHIBITION
    # ("不得写「已实现 SOTA」", "禁写：…", "新增两条不可再写：").  Matching the
    # bare substring would flag the rule itself as a violation, so the
    # context in front of the phrase is inspected: a prohibition marker
    # before it (on the same line, or in the two preceding lines that
    # introduce the prohibition) means it is NOT an assertion.
    print("\n  -- forbidden claims absent as assertions --")
    PROHIBITION = ("禁写", "不得写", "不能写", "不可写", "不可再写", "不得再写",
                   "不得宣称", "不可宣称", "不得用", "不可用")
    for key, text in texts.items():
        lines = text.splitlines()
        violations = []
        for i, line in enumerate(lines):
            for phrase in FORBIDDEN:
                pos = line.find(phrase)
                if pos < 0:
                    continue
                prefix = line[:pos]
                prev = "\n".join(lines[max(0, i - 2):i])
                if any(m in prefix for m in PROHIBITION):
                    continue          # prohibition on the same line
                if any(m in prev for m in PROHIBITION):
                    continue          # prohibition introduced just above
                violations.append((i + 1, phrase))
        if violations:
            bad(f"{key}: forbidden claim asserted at line(s) "
                f"{[v[0] for v in violations]}: {[v[1] for v in violations]}")
        else:
            ok(f"{key}: no forbidden claim asserted "
               f"(occurrences inside prohibitions are ignored)")

    # anchor numbers: informational (a doc may legitimately omit a probe)
    print("\n  -- anchor numbers (informational) --")
    anchors = ["0.0924", "0.1653", "402.43", "0.5526", "0.9712",
               "0.6103", "0.4035", "2.0040", "2.1124"]
    for key, text in texts.items():
        present = [a for a in anchors if a in text]
        missing = [a for a in anchors if a not in text]
        info(f"{key}: {len(present)}/{len(anchors)} anchors present"
             + (f"; absent: {missing}" if missing else ""))


# --------------------------------------------------------------------------
# C. the three registered implementation defects: located vs implemented
# --------------------------------------------------------------------------
def check_defects():
    print("\n=== C. implementation status of the three registered defects ===")
    topo = (ROOT / "models" / "topo_simvp" / "model.py")
    scaled = (ROOT / "models" / "scaled_structure" / "model.py")
    probe = (ROOT / "tools" / "_e45_mechanism_probe.py")

    d1_fixed = topo.exists() and "clamp_mm" in topo.read_text(encoding="utf-8")
    if d1_fixed:
        warn("D1 amplitude truncation: 'clamp_mm' now reaches the output field "
             "-> E45 must be re-measured, docs are stale")
    else:
        ok("D1 amplitude truncation: still UNFIXED "
           "(no clamp_mm entry in models/topo_simvp/model.py)")

    d2_fixed = False
    if scaled.exists():
        txt = scaled.read_text(encoding="utf-8")
        d2_fixed = bool(re.search(r"_SHAPE_EPS\s*=\s*(?!1e-12\b)[0-9.e\-]+", txt))
    if d2_fixed:
        warn("D2 shape normalisation guard: no longer the bare 1e-12 "
             "-> E45 must be re-measured, docs are stale")
    else:
        ok("D2 shape normalisation guard: still the absolute 1e-12 "
           "(models/scaled_structure/model.py)")

    d3_fixed = False
    if probe.exists():
        txt = probe.read_text(encoding="utf-8")
        d3_fixed = "np.nanmedian(" in txt
    if d3_fixed:
        warn("D3 probe aggregation: now uses np.nanmedian -> the registered "
             "maxratio_as_trained > 0.9 criterion must be re-read")
    else:
        ok("D3 probe aggregation: still np.median (non nan-aware) -> UNFIXED")

    n_fixed = sum([d1_fixed, d2_fixed, d3_fixed])
    info(f"implementation status: {n_fixed}/3 fixed "
         f"(fix located != fix implemented)")


# --------------------------------------------------------------------------
# D. reporting-shape check: median + pass rate, not single-seed scores
# --------------------------------------------------------------------------
def check_reporting_shape():
    print("\n=== D. scores_summary reporting shape ===")
    path = ROOT / "tools" / "_evals" / "e45" / "scores_summary.json"
    if not path.exists():
        bad("scores_summary.json missing locally")
        return
    data = json.loads(path.read_text(encoding="utf-8"))

    keys = set()

    def walk(o):
        if isinstance(o, dict):
            keys.update(str(k).lower() for k in o.keys())
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    has_median = any("median" in k or "中位" in k for k in keys)
    has_rate = any(("rate" in k or "pass" in k or "frac" in k or "通过" in k)
                   for k in keys)
    if has_median:
        ok("summary carries a median read-out")
    else:
        bad("summary carries NO median read-out (reporting rule violated)")
    if has_rate:
        ok("summary carries a pass-rate read-out")
    else:
        bad("summary carries NO pass-rate read-out (reporting rule violated)")


def main():
    print("=" * 78)
    print("E45 round-34 verification (read-only)")
    print("=" * 78)
    check_s3()
    check_semantics()
    check_defects()
    check_reporting_shape()
    print("\n" + "=" * 78)
    if FAILS:
        print(f"FAIL count = {len(FAILS)}")
        for f in FAILS:
            print(f"  - {f}")
        print("VERDICT: FAIL")
        return 1
    if WARNS:
        print(f"WARN count = {len(WARNS)} (source changed -> re-measurement due)")
        for w in WARNS:
            print(f"  - {w}")
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
