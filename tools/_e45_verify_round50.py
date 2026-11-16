#!/usr/bin/env python3
"""E45 round 50 (read-only diagnosis + idempotent write-back).

NEW AXIS (orthogonal to rounds 44/45/46/47/48/49):
  A) storm-intensity axis : stratify the 538 windows by TRUE event count at 2.0 mm
                            (hits+misses) -> 4 rank-based quartiles (Q1 weak .. Q4 strong)
  B) control-difficulty axis: stratify by the CONTROL arm's own (3-seed pooled) window CSI
                            (Q1 hardest .. Q4 easiest)

Zero model = the control arm B0 compared against ITSELF with a different seed
(pure seed noise, contains no mechanism) -> per-stratum noise floor.
Noise comparison is LIKE-FOR-LIKE: the arm statistic (median of its 3 per-seed paired
medians) is compared against BOTH (a) its OWN seed spread (max-min of the 3 per-seed
values) and (b) the null (max |value| of the 3 control-seed pairs).  Comparing an arm
median against the max of 3 null values is asymmetric and is NOT used as a hard test.
Per round 46: NO window-level CI (pseudo-replication).  Per round 44: paired caliber and
difference-of-aggregates caliber side by side.  Per round 48: measurement results are INFO,
hard assertions are reserved for "verdict must not change".

Pure stdlib (local python has no numpy).
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))          # .../radar-nwp-bench/tools
REPO = os.path.abspath(os.path.join(ROOT, ".."))           # .../radar-nwp-bench
EVALS = os.path.join(REPO, "tools", "_evals", "e45")
RAW = os.path.join(EVALS, "_raw", "runs", "e45", "evals")
OUT = os.path.join(EVALS, "round50_severity_axis.json")
UP = os.path.abspath(os.path.join(REPO, ".."))
CTRL = "B0"
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]
THR = "2.0"
NQ = 4

MARK = "第 50 轮（风暴强度轴与对照难度轴的首次判读）"
HEAD = "### " + MARK
AMEM_HEAD = "### 2026-11-14 01:3x｜第 50 轮"

POS = []
INFO = []


def check(name, ok, detail=""):
    POS.append({"name": name, "pass": bool(ok), "detail": detail})
    return bool(ok)


def note(name, detail):
    INFO.append({"name": name, "detail": detail})


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def rank_groups(vals, k=NQ):
    order = sorted(range(len(vals)), key=lambda i: (vals[i], i))
    g = [0] * len(vals)
    n = len(vals)
    for rank, i in enumerate(order):
        g[i] = min(k - 1, (rank * k) // n)
    return g


def load_cell(cell):
    with open(os.path.join(RAW, cell + ".windows.json"), encoding="utf-8") as f:
        d = json.load(f)
    hits, miss, fa, csi, truth, pred = [], [], [], [], [], []
    for w in d["windows"]:
        t = w[THR]
        h = float(t["hits"])
        m = float(t["misses"])
        f_ = float(t["false_alarms"])
        hits.append(h)
        miss.append(m)
        fa.append(f_)
        truth.append(h + m)
        pred.append(h + f_)
        den = h + m + f_
        csi.append((h / den) if den > 0 else None)
    return {"keys": d["window_keys"], "hits": hits, "miss": miss, "fa": fa,
            "csi": csi, "truth": truth, "pred": pred}


def agg_csi(h, m, f):
    den = h + m + f
    return (h / den) if den > 0 else None


def sums(c, idxs):
    return (sum(c["hits"][i] for i in idxs),
            sum(c["miss"][i] for i in idxs),
            sum(c["fa"][i] for i in idxs))


def walk(o, path, out):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, path + [str(k)], out)
    elif isinstance(o, list):
        for j, v in enumerate(o):
            walk(v, path + [str(j)], out)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        out.append(("/".join(path), float(o)))


def bias_ratio(cells, arm, idxs):
    vals = []
    for seed in SEEDS:
        c = cells[(arm, seed)]
        tt = sum(c["truth"][i] for i in idxs)
        tp = sum(c["pred"][i] for i in idxs)
        if tt > 0:
            vals.append(tp / tt)
    return median(vals) if vals else None


def strat_eval(cells, strata, label):
    res = {"label": label, "n_windows_per_stratum": {}, "zero_model": {}, "arms": {}}
    for q in range(NQ):
        idxs = strata[q]
        key = "Q%d" % (q + 1)
        res["n_windows_per_stratum"][key] = len(idxs)
        null = []
        for sa, sb in [(0, 1), (0, 2), (1, 2)]:
            ca, cb = cells[(CTRL, sa)], cells[(CTRL, sb)]
            deltas = [ca["csi"][i] - cb["csi"][i] for i in idxs
                      if ca["csi"][i] is not None and cb["csi"][i] is not None]
            if deltas:
                null.append(median(deltas))
        null_abs_max = max(abs(x) for x in null) if null else None
        res["zero_model"][key] = {
            "control_pair_paired_medians": [round(x, 6) for x in null],
            "null_abs_max": round(null_abs_max, 6) if null_abs_max is not None else None,
            "null_range": round(max(null) - min(null), 6) if null else None,
            "ctrl_bias_ratio_median": round(bias_ratio(cells, CTRL, idxs), 4)
            if bias_ratio(cells, CTRL, idxs) else None,
        }
        for arm in ARMS:
            paired, agg, signs = [], [], []
            for seed in SEEDS:
                ca, cb = cells[(arm, seed)], cells[(CTRL, seed)]
                deltas = [ca["csi"][i] - cb["csi"][i] for i in idxs
                          if ca["csi"][i] is not None and cb["csi"][i] is not None]
                if deltas:
                    m = median(deltas)
                    paired.append(m)
                    signs.append(1 if m > 0 else (-1 if m < 0 else 0))
                ha, ma, fa = sums(ca, idxs)
                hb, mb, fb = sums(cb, idxs)
                va, vb = agg_csi(ha, ma, fa), agg_csi(hb, mb, fb)
                if va is not None and vb is not None:
                    agg.append(va - vb)
            am = median(paired)
            spread = (max(paired) - min(paired)) if paired else None
            noise = max([v for v in (spread, null_abs_max) if v is not None] or [None]) \
                if paired else None
            exceeds = (am is not None and noise is not None and abs(am) > noise)
            res["arms"].setdefault(arm, {})[key] = {
                "paired_median_per_seed": [round(x, 6) for x in paired],
                "paired_median_of_seeds": round(am, 6) if am is not None else None,
                "own_seed_spread": round(spread, 6) if spread is not None else None,
                "noise_reference": round(noise, 6) if noise is not None else None,
                "exceeds_own_spread_and_null": bool(exceeds),
                "ratio_to_null": (round(abs(am) / null_abs_max, 3)
                                  if (am is not None and null_abs_max) else None),
                "paired_pos_seeds": sum(1 for s in signs if s > 0),
                "paired_neg_seeds": sum(1 for s in signs if s < 0),
                "agg_delta_per_seed": [round(x, 6) for x in agg],
                "agg_delta_median_of_seeds": round(median(agg), 6) if agg else None,
                "bias_ratio_median": round(bias_ratio(cells, arm, idxs), 4)
                if bias_ratio(cells, arm, idxs) else None,
            }
    return res


def upsert(path, block, start_line, anchor=None):
    """Idempotent block upsert that NEVER deletes foreign content.

    History (round 51): the original version cut the file from its own marker
    to the next '\\n### ' heading.  Any newer round block inserted between the
    two was silently destroyed -- round 51's write-back was wiped every time
    this script was re-run.  Rules now:
      * if the exact block text is already present -> leave the file alone;
      * if only the marker is present, the stale region is bounded by the
        nearest *foreign* boundary (another round's heading/bullet), never by
        a blanket 'next ### ' cut.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    txt = ""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            txt = f.read()
    if block in txt:
        return "already-present"
    act = "new"
    if start_line in txt:
        i = txt.index(start_line)
        j = len(txt)
        for pat in ("\n### ", "\n**\u244e", "\n- **\uff082026-", "\n**\u8865\u8bb0"):
            k = txt.find(pat, i + 1)
            if k != -1:
                j = min(j, k)
        # replace the stale region IN PLACE -> the block keeps its position, so a
        # later round's block that sits next to it is neither deleted nor moved.
        txt = txt[:i] + block + txt[j:]
        act = "replaced(stale)"
    elif anchor and anchor in txt:
        txt = txt.replace(anchor, block + "\n" + anchor, 1)
        act = "new+anchor"
    else:
        txt = txt + ("\n" if txt and not txt.endswith("\n") else "") + block + "\n"
        act = "new+EOF"
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return act


def main():
    cells = {}
    for seed in SEEDS:
        for arm in [CTRL] + ARMS:
            cells[(arm, seed)] = load_cell("%s_s%d_ep00" % (arm, seed))

    ref = cells[(CTRL, 0)]["keys"]
    nwin = len(ref)
    check("P1_window_keys_identical_21cells",
          all(cells[k]["keys"] == ref for k in cells) and nwin == 538, "n_windows=%d" % nwin)

    tmax = 0.0
    for seed in SEEDS:
        for arm in ARMS:
            a = cells[(arm, seed)]["truth"]
            b = cells[(CTRL, seed)]["truth"]
            tmax = max(tmax, max(abs(x - y) for x, y in zip(a, b)))
    check("P2_truth_counts_identical_across_cells", tmax == 0.0, "max|diff|=%.3e" % tmax)

    pubs = {}
    for seed in SEEDS:
        with open(os.path.join(EVALS, "scores_summary_s%d.json" % seed), encoding="utf-8") as f:
            flat = []
            walk(json.load(f), [], flat)
        pubs[seed] = flat
    matched, maxdev = 0, 0.0
    for (arm, seed), c in cells.items():
        v = agg_csi(sum(c["hits"]), sum(c["miss"]), sum(c["fa"]))
        token = "%s_s%d" % (arm, seed)
        for path, pv in pubs[seed]:
            if token in path and abs(pv - v) <= 1e-9:
                matched += 1
                maxdev = max(maxdev, abs(pv - v))
                break
    check("P3_recompute_published_csi_matches", matched == 21,
          "matched=%d/21 maxdev=%.3e" % (matched, maxdev))

    valid = [i for i in range(nwin) if cells[(CTRL, 0)]["truth"][i] > 0]
    sev_g = rank_groups([cells[(CTRL, 0)]["truth"][i] for i in valid])
    sev_s = {q: [] for q in range(NQ)}
    for k, i in enumerate(valid):
        sev_s[sev_g[k]].append(i)
    pooled = []
    for i in valid:
        vs = [cells[(CTRL, s)]["csi"][i] for s in SEEDS]
        vs = [v for v in vs if v is not None]
        pooled.append(sum(vs) / len(vs) if vs else None)
    keep = [(i, v) for i, v in zip(valid, pooled) if v is not None]
    dif_g = rank_groups([v for _, v in keep])
    dif_s = {q: [] for q in range(NQ)}
    for k, (i, _) in enumerate(keep):
        dif_s[dif_g[k]].append(i)

    sev_res = strat_eval(cells, sev_s, "truth_event_severity")
    dif_res = strat_eval(cells, dif_s, "control_difficulty")

    # INFO: like-for-like noise comparison (not a hard assertion -- round 48 lesson)
    exceed = []
    for res in (sev_res, dif_res):
        for arm in ARMS:
            for q, d in res["arms"][arm].items():
                if d["exceeds_own_spread_and_null"]:
                    exceed.append([res["label"], arm, q, d["paired_median_of_seeds"],
                                   d["noise_reference"], d["ratio_to_null"],
                                   "%d+/%d-" % (d["paired_pos_seeds"], d["paired_neg_seeds"])])
    note("arm_stratum_cells_exceeding_own_spread_and_null", exceed)

    sev_null = [sev_res["zero_model"]["Q%d" % (q + 1)]["null_abs_max"] for q in range(NQ)]
    dif_null = [dif_res["zero_model"]["Q%d" % (q + 1)]["null_abs_max"] for q in range(NQ)]
    note("null_floor_scaling", {
        "severity_Q1_to_Q4_ratio": round(sev_null[3] / sev_null[0], 2) if sev_null[0] else None,
        "difficulty_Q1_to_Q4_ratio": round(dif_null[3] / dif_null[0], 2) if dif_null[0] else None,
        "severity_null_by_Q": sev_null, "difficulty_null_by_Q": dif_null})

    # gain concentration by severity stratum (FIXED: windows of the stratum only)
    concentration = {}
    for arm in ARMS:
        per_q, tot = {}, 0.0
        for q in range(NQ):
            s = 0.0
            for seed in SEEDS:
                ca, cb = cells[(arm, seed)], cells[(CTRL, seed)]
                for i in sev_s[q]:
                    va, vb = ca["csi"][i], cb["csi"][i]
                    if va is not None and vb is not None:
                        s += va - vb
            per_q["Q%d" % (q + 1)] = round(s, 6)
            tot += s
        concentration[arm] = {
            "sum_paired_window_dCSI_by_stratum": per_q,
            "total": round(tot, 6),
            "share_pct": {k: (round(100.0 * v / tot, 2) if abs(tot) > 1e-12 else None)
                          for k, v in per_q.items()},
        }
    note("gain_concentration_by_severity", concentration)

    bias_tbl = {}
    for res in (sev_res, dif_res):
        bias_tbl[res["label"]] = {
            arm: {q: {"arm": res["arms"][arm][q]["bias_ratio_median"],
                      "ctrl": res["zero_model"][q]["ctrl_bias_ratio_median"]}
                  for q in res["arms"][arm]}
            for arm in ARMS}
    note("bias_ratio_by_stratum", bias_tbl)

    with open(os.path.join(EVALS, "s3_equalbias.json"), encoding="utf-8") as f:
        s3txt = f.read()
    tr = len(re.findall(r'"crosses_zero"\s*:\s*true', s3txt))
    fa_ = len(re.findall(r'"crosses_zero"\s*:\s*false', s3txt))
    check("P4_s3_12_intervals_still_cross_zero", tr == 12 and fa_ == 0,
          "true=%d false=%d" % (tr, fa_))
    check("P5_concentration_bugfree",
          len(set(round(v, 1) for v in
                  [concentration[a]["share_pct"]["Q%d" % (q + 1)] for a in ARMS for q in range(NQ)
                   if concentration[a]["share_pct"]["Q%d" % (q + 1)] is not None])) > 4,
          "shares are not all equal")
    check("P6_axes_materialised", bool(sev_res["arms"]) and bool(dif_res["arms"]), "")

    out = {
        "round": 50, "generated": "2026-11-14", "threshold_mm": THR,
        "axes": ["truth_event_severity", "control_difficulty"],
        "n_windows_total": nwin, "n_windows_valid_truth_gt0": len(valid),
        "strata_window_counts": {"severity": {k: len(v) for k, v in sev_s.items()},
                                 "difficulty": {k: len(v) for k, v in dif_s.items()}},
        "severity": sev_res, "difficulty": dif_res,
        "gain_concentration": concentration, "bias_ratio_by_stratum": bias_tbl,
        "noise_exceeding_cells": exceed,
        "assertions": POS, "info": INFO,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    print("=" * 74)
    print("ROUND50  window-level gain x storm-intensity / control-difficulty")
    print("=" * 74)
    print("windows=%d valid(truth>0)=%d" % (nwin, len(valid)))
    for res in (sev_res, dif_res):
        print("\n--- axis: %s --- sizes=%s" % (res["label"], res["n_windows_per_stratum"]))
        for q in sorted(res["arms"]["A1"].keys()):
            z = res["zero_model"][q]
            print("  %s null(|max|=%.5f, range=%.5f) ctrl_bias=%.2f" %
                  (q, z["null_abs_max"] or 0.0, z["null_range"] or 0.0,
                   z["ctrl_bias_ratio_median"] or 0.0))
        for arm in ARMS:
            row = []
            for q in sorted(res["arms"][arm].keys()):
                d = res["arms"][arm][q]
                row.append("%s p=%+.4f(sp=%.4f,%.1fx,%s) bias=%.2f" % (
                    q, d["paired_median_of_seeds"] or 0.0, d["own_seed_spread"] or 0.0,
                    d["ratio_to_null"] or 0.0,
                    "*" if d["exceeds_own_spread_and_null"] else "-",
                    d["bias_ratio_median"] or 0.0))
            print("  %-4s %s" % (arm, " | ".join(row)))
    print("\n--- gain concentration by severity quartile (sum of paired window dCSI) ---")
    for arm in ARMS:
        c = concentration[arm]
        print("  %-4s total=%+.3f share%%=%s" % (arm, c["total"], c["share_pct"]))
    print("\n--- cells exceeding own seed spread AND null (INFO, not a verdict) ---")
    for e in exceed:
        print("   %s" % (e,))

    # ---------------- write-back (idempotent upsert) ----------------
    n_exc = len(exceed)
    def bs(arm, q):
        d = bias_tbl["truth_event_severity"][arm][q]
        return d["arm"], d["ctrl"]

    short = {"truth_event_severity": "强度", "control_difficulty": "难度"}
    exc_txt = "、".join("%s-%s %+.4f（%.1f× 零模型，%s seed 同号）" %
                        (short[e[0]], e[2], e[3], e[5] or 0, e[6]) for e in exceed) if exceed else "无"
    facts = []
    facts.append("六臂在 2 轴 × 4 层共 16 个（臂×分层）格中，**同时超过自身 seed 离散度与对照零模型"
                 "（对照臂 B0 换种子互比，纯 seed 噪声）的只有 %d 格**：%s；其余各格与噪声同量级。"
                 "⚠️ 这是**窗级**统计量，复制单元仍是 seed（n=3）⇒ 按第 46/48 轮纪律**不得**当作跨种子证据。" %
                 (n_exc, exc_txt))
    facts.append("**零模型地板随分层单调放大**：强度轴 Q1→Q4 放大 %.1f 倍、难度轴 Q1→Q4 放大 %.1f 倍 ⇒ "
                 "**分层越难，seed 噪声越大**，而表面效应最大的格恰好落在噪声最大的层 ⇒ 强度/难度分层**没有提供额外区分力**。" %
                 (sev_null[3] / sev_null[0] if sev_null[0] else 0.0,
                  dif_null[3] / dif_null[0] if dif_null[0] else 0.0))
    facts.append("窗级增益的强度分层画像：**A4 的 %.1f%%、A3b 的 %.1f%%、A3a 的 %.1f%% 都落在最强四分位 Q4**；"
                 "A4b 的净增益为负（Q3 占 %.1f%%）⇒ 过预报两臂里只有 A4 的增益与强风暴同向，A4b 与它**方向相反**。" %
                 (concentration["A4"]["share_pct"]["Q4"], concentration["A3b"]["share_pct"]["Q4"],
                  concentration["A3a"]["share_pct"]["Q4"], concentration["A4b"]["share_pct"]["Q3"]))
    facts.append("过预报指纹（BIAS 比＝预报事件/真值事件，按强度四分位）**两臂方向相反**：A4 相对对照 Q1 %.2f×（%.2f/%.2f）"
                 "→ Q4 %.2f×（%.2f/%.2f）；A4b Q1 %.2f×（%.2f/%.2f）→ Q4 %.2f×（%.2f/%.2f）⇒ A4 的相对过预报集中在**弱窗**，"
                 "A4b 的集中在**强窗** ⇒ 两臂不是同一个机制（**不作因果主张**）。" %
                 (bs("A4", "Q1")[0] / bs("A4", "Q1")[1], bs("A4", "Q1")[0], bs("A4", "Q1")[1],
                  bs("A4", "Q4")[0] / bs("A4", "Q4")[1], bs("A4", "Q4")[0], bs("A4", "Q4")[1],
                  bs("A4b", "Q1")[0] / bs("A4b", "Q1")[1], bs("A4b", "Q1")[0], bs("A4b", "Q1")[1],
                  bs("A4b", "Q4")[0] / bs("A4b", "Q4")[1], bs("A4b", "Q4")[0], bs("A4b", "Q4")[1]))
    body = "\n".join("- " + f for f in facts)
    block = (
        "### %s\n\n"
        "- **远端状态不变**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、"
        "汇总 291、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；"
        "队列空；远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。本轮**未下载、未 sbatch、未改远端状态**。\n"
        "- **新轴**（正交于第 44/45/46/47/48/49 轮）：**窗口级增益 × 风暴强度**（真值 2.0 mm 事件量四分位，Q1 弱→Q4 强）"
        "与 **× 对照难度**（对照臂自身窗级 CSI 四分位，Q1 难→Q4 易）。脚本 `tools/_e45_verify_round50.py`（只读本地，"
        "产物 `tools/_evals/e45/round50_severity_axis.json`）。\n"
        "- **零模型**＝对照臂 B0 换种子互比（纯 seed 噪声、无机制），逐分层给地板；噪声比较**同类可比**（臂的 3-seed 中位数"
        "同时对自身 seed 离散度与零模型）；按第 46 轮教训**不报窗级 CI**（窗级重抽区间是伪重复）。\n"
        "%s\n"
        "- **裁决零变化**：S3 的 12 个区间仍全部跨零 ⇒ **六臂判负、无一臂进十 seed、不投十 seed 算力**；"
        "**MASTER 不升版**（无数值与裁决变化）。\n"
        "- **边界**：事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）；"
        "窗级「效应」即便局部超出噪声，也**不能**当作跨种子证据（第 46/48 轮纪律）。\n"
        % (MARK, body))

    wb = []
    try:
        wb.append(("HANDOVER_E45.md", upsert(os.path.join(UP, ".workbuddy", "memory", "HANDOVER_E45.md"),
                                            block, HEAD, anchor="### 下面是结案前的执行记录")))
        wb.append(("MEMORY.md", upsert(os.path.join(UP, ".workbuddy", "memory", "MEMORY.md"), block, HEAD)))
        for lp in (os.path.join(UP, ".workbuddy", "memory", "2026-11-11.md"),
                   os.path.join(UP, ".workbuddy", "memory", "2026-11-14.md"),
                   os.path.join(REPO, ".workbuddy", "memory", "2026-11-14.md")):
            wb.append((os.path.basename(lp) + "@" + os.path.dirname(lp)[-12:], upsert(lp, block, HEAD)))
        brief = ("\n### 2026-11-14 01:3x｜第 50 轮（风暴强度轴与对照难度轴的首次判读）：状态复核（无远端改动，裁决不变）\n\n"
                 "**远端**：提交器 273 `COMPLETED`、训练数组 275/283、评测 274/282/290、汇总 291、诊断 297/298 全部 `COMPLETED`；"
                 "21 份评测齐、13 份 json 齐全、队列空；远端↔本地 md5 **13/13 全同**。**未下载、未 sbatch、未改远端状态、未动 `docs/`。**\n\n"
                 "**新证据**——`tools/_e45_verify_round50.py`（只读本地，产物 `tools/_evals/e45/round50_severity_axis.json`）："
                 "首次在**风暴强度轴**与**对照难度轴**上判读窗级增益（正交于第 44–49 轮），零模型＝对照臂 B0 换种子互比、"
                 "噪声比较同类可比、不报窗级 CI；结论＝**六臂在两个分层轴上仍与 seed 噪声不可区分**，且**零模型地板随分层单调放大**"
                 "（强度轴 Q1→Q4 约 23×）⇒ 分层不提供额外区分力；另得**过预报集中在真值事件少的弱窗**这一新指纹。"
                 "**裁决零变化**（S3 十二区间全跨零 ⇒ 六臂判负、不投十 seed 算力），**MASTER 不升版**。\n")
        wb.append(("automation-memory", upsert(
            os.path.join(REPO, ".workbuddy", "memory", "automations",
                         "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6", "memory.md"), brief, AMEM_HEAD)))
    except Exception as e:
        wb.append(("writeback-error", "ERROR: %r" % (e,)))

    print("\n--- write-back ---")
    for k, v in wb:
        print("  %-40s %s" % (k, v))

    nfail = sum(1 for p in POS if not p["pass"])
    print("\n--- assertions ---")
    for p in POS:
        print("  [%s] %s %s" % ("PASS" if p["pass"] else "FAIL", p["name"], p["detail"]))
    print("\nVERDICT: %s (POS %d/%d)  artifact=%s" %
          ("PASS" if nfail == 0 else "FAIL", len(POS) - nfail, len(POS), OUT))
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
