"""Round-44 bookkeeping: register the new multi-threshold evidence.

What this records
-----------------
Round 44 added one genuinely new piece of evidence about the already-closed
E45 experiment: the six arms' skill profile across THREE thresholds
(0.5 / 2.0 / 4.0 mm), read under TWO different "median gain" definitions
(paired per-seed median difference vs. difference of medians), plus the
POD/FAR/BIAS decomposition and the gain-vs-over-forecast covariance.

Why it matters (and why the verdict does NOT move)
--------------------------------------------------
The pre-registered S3 guard sits at thr=2.0 mm and S5 forbids switching
thresholds, so this is POST-HOC DIAGNOSTIC, not a new gate:
  * no arm is better than the control at the light-rain threshold (0.5 mm);
  * the only arms that beat the control ARM-LEVEL at 2.0 / 4.0 mm are A4 and
    A4b -- exactly the two arms whose POD and BIAS blow up;
  * A3a / A3b look positive only under the paired reading; under the
    arm-level reading they are WORSE than the control at 0.5 / 2.0 mm, i.e.
    their apparent gain is a seed-alignment artifact.
This independently corroborates the frozen S3 verdict ("still an
area/amplitude effect") from a different axis, and changes no verdict.

New reusable reading rule (registered next to §4-④):
  the paired median gain and the difference-of-medians MUST be reported
  side by side; reporting only the paired one can turn seed alignment into
  an apparent mechanism gain.

Idempotent: running it twice changes nothing.  Markers are exact substrings
of the inserted text (the round-41 / round-43 lesson).
"""
from pathlib import Path
import sys

REPO = Path("R:/1radarNWPprediction/radar-nwp-bench")
HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
DAILY_UP = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-13.md")
DAILY_PJ = REPO / ".workbuddy/memory/2026-11-13.md"
LOG_1111 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")
AUTOMATION = REPO / ".workbuddy/memory/automations/e3cc6636-3e60-4a58-8f7e-67d4c896b0b6/memory.md"

MARK = "第 44 轮"

# --------------------------------------------------------------------------
# 1) HANDOVER_E45.md  §4⑨  (inserted before the "结案前的执行记录" archive)
# --------------------------------------------------------------------------
HANDOVER_ITEM = """
**⑨（2026-11-13 18:2x 第 44 轮新增证据）多阈值技能剖面 ＋ 两种「中位数增益」口径的分离**——脚本 `tools/_e45_verify_round44.py`（只读本地，**VERDICT: PASS**）：已取回的产物里每格都带 `csi/bias` @0.5/2.0/4.0 三个阈值，但**前 43 轮的判读全部只在 thr=2.0 上做**。本轮第一次把六臂摊到三个阈值，并把两种口径并列：
- **配对口径**＝三 seed 逐 seed 差值取中位数；**差值口径**＝两条中位数相减（臂级真实水平）。
- **弱降雨阈值（0.5 mm）上没有任何一臂优于对照**：差值口径六臂全负（A3a −0.0245 最接近 0，A4 −0.0315、A4b −0.0140、A2 −0.0174、A1 −0.0015、A3b −0.0009）。
- **臂级（差值口径）真正优于对照的只有 A4 与 A4b**（thr2.0 +0.0221／+0.0157，thr4.0 +0.0311／+0.0313）——而这正是 POD 与 BIAS 同时爆表的两臂（thr2.0 中位数：A4b POD 0.4692／BIAS 2.8215，A4 POD 0.2222／BIAS 0.8362，对照 B0 POD 0.1639／BIAS 0.5095）⇒ **与 S3「仍是面积／幅度效应」从不同轴独立吻合**。
- **A3a／A3b 的「正增益」只在配对口径下成立**：差值口径在 0.5／2.0 mm 上均为负（A3a −0.0245／−0.0118，A3b −0.0009／−0.0058）⇒ 那两臂的表面增益来自**种子对齐效应**（臂的好种子对上对照的坏种子），不是臂级真实水平；两臂早已分别被拓扑门（`grad_top1pct_share` ≥ 0.5、`field_max_mm` 402.43 > 30）与 S3 判负 ⇒ **不重开**。
- **增益与正偏置共变且随阈值增强**：r(dCSI, dBIAS) = +0.4959(0.5)／+0.5576(2.0)／+0.6800(4.0)（n=18）⇒ 越「涨分」的格越「过预报」。
- **口径纪律（新增，可复用）**：**配对中位数增益与差值口径必须并列报告**；只报配对口径会把种子对齐效应当成机制增益（与 §4④「过地板通过率不能单独用于晋级」并列的同族规则）。
- **纪律边界**：本条是**事后诊断**，S3 护栏按预注册固定在 thr=2.0、S5 明令「不换阈值」⇒ **不构成新判据、不改变任何裁决**，0.5/4.0 mm 上亦**无** bootstrap CI（bootstrap 只在 2.0 mm 做过）⇒ 只报中位数与通过率。
"""

HANDOVER_ANCHOR = "---\n\n### 下面是结案前的执行记录（存档用，不必重跑）"

# --------------------------------------------------------------------------
# 2) MEMORY.md  状态段
# --------------------------------------------------------------------------
MEMORY_ANCHOR = ("- **地板口径不稳**（地板 0.0924 是单一种子口径，对照臂 B0 自己只 2/3 过）"
                 "⇒ **过地板通过率不能单独用于晋级判定**。")
MEMORY_NEW = (
    MEMORY_ANCHOR + "\n"
    "- **两种「中位数增益」口径必须并列（第 44 轮新增，同族规则）**：配对口径（逐 seed 差值的中位数）"
    "会因**种子对齐**把对照的坏种子配给臂的好种子而虚高；E45 六臂里 **A3a／A3b 只在配对口径下为正**，"
    "差值口径（臂级真实水平）在 0.5／2.0 mm 上均为负 ⇒ 臂级真正优于对照的只有 A4／A4b，"
    "而这两臂正是 POD／BIAS 爆表者 ⇒ 与 S3「仍是面积／幅度效应」独立吻合（`tools/_e45_verify_round44.py`）。"
    "**裁决不变：六臂判负、不投十 seed 算力。**"
)

# --------------------------------------------------------------------------
# 3) 逐日日志（三份：上层今日、项目今日、2026-11-11 追记）
# --------------------------------------------------------------------------
DAILY_NOTE = (
    "\n\n- **（2026-11-13 18:2x，第 44 轮）** E45 结案结论不变"
    "（六臂判负、无一臂进十 seed、不投十 seed 算力）；**新证据＝多阈值技能剖面 ＋ 两种中位数增益口径的分离**"
    "（`tools/_e45_verify_round44.py`，只读本地，VERDICT: PASS）："
    "① 前 43 轮的判读全部只在 thr=2.0 上做，本轮第一次把六臂摊到 0.5/2.0/4.0 mm；"
    "② **弱降雨阈值（0.5 mm）上差值口径六臂全负**——没有任何一臂改善弱降雨技能；"
    "③ **臂级真正优于对照的只有 A4／A4b**（thr2.0 +0.0221／+0.0157，thr4.0 +0.0311／+0.0313），"
    "而这两臂正是 POD／BIAS 爆表者（A4b POD 0.4692／BIAS 2.8215 对 B0 的 0.1639／0.5095）"
    "⇒ **与 S3「仍是面积/幅度效应」从不同轴独立吻合**；"
    "④ **A3a／A3b 的正增益只在配对口径下成立**，差值口径在 0.5／2.0 mm 为负 ⇒ 属**种子对齐效应**，不重开；"
    "⑤ r(dCSI, dBIAS) = +0.4959／+0.5576／+0.6800（0.5/2.0/4.0 mm）⇒ 越涨分越过预报；"
    "⑥ **新口径纪律**：配对口径与差值口径必须并列报告。"
    "**边界**：事后诊断，S3 护栏固定在 thr=2.0、S5 不换阈值 ⇒ 不构成新判据、不改变任何裁决。\n"
)

LOG1111_NOTE = (
    "\n- **（2026-11-13 18:2x 补记，第 44 轮）** E45 结案结论不变"
    "（六臂判负、不投十 seed 算力）；新证据＝多阈值技能剖面与两种中位数增益口径的分离："
    "**弱降雨阈值 0.5 mm 上差值口径六臂全负**；**臂级真正优于对照的只有 A4／A4b**"
    "（而这两臂 POD／BIAS 同时爆表 ⇒ 与 S3「仍是面积/幅度效应」独立吻合）；"
    "**A3a／A3b 的正增益只在配对口径成立、差值口径为负 ⇒ 种子对齐效应**，不重开。"
    "详见 `HANDOVER_E45.md` §4⑨ 与 `tools/_e45_verify_round44.py`。\n"
)

AUTOMATION_NOTE = (
    "\n\n### 2026-11-13 18:2x｜第 44 轮：状态复核（无远端改动，裁决不变）"
    "＋ 新证据＝多阈值技能剖面与两种「中位数增益」口径的分离\n\n"
    "**远端**：提交器 **273 COMPLETED**（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、"
    "汇总 291、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份**评测齐；"
    "`runs/e45/` **13 份** json 齐全；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；"
    "队列仅剩 1 个 `qlfn2-curve`（304，**非 E45，不干预**）。**本轮未下载、未 sbatch、未改远端状态。**\n\n"
    "**产物一致性**：远端 `runs/e45/*.json` 与本地 `tools/_evals/e45/*.json` md5 **13/13 全同**。\n\n"
    "**闸门**：`_check_tables` PASS（`introduced=[]`）、`_check_v245/246/247/248_row` 全 PASS、"
    "`_e45_docs_numbers` PASS、`_e45_verify_round40/41/42/43` 全 PASS ⇒ **MASTER 仍 v2.48、裁决不变**。\n\n"
    "**本轮新东西（前 43 轮都只在 thr=2.0 上判读）**——新脚本 `tools/_e45_verify_round44.py`"
    "（只读本地，VERDICT: PASS）：① 把六臂摊到 **0.5/2.0/4.0 mm** 三个阈值；"
    "② 并列**配对口径**（逐 seed 差值中位数）与**差值口径**（中位数相减）；"
    "③ **0.5 mm 上差值口径六臂全负**（无臂改善弱降雨）；"
    "④ **臂级真正优于对照的只有 A4／A4b**（thr2.0 +0.0221／+0.0157，thr4.0 +0.0311／+0.0313），"
    "而这两臂 POD／BIAS 同时爆表 ⇒ **与 S3「仍是面积/幅度效应」从不同轴独立吻合**；"
    "⑤ **A3a／A3b 的正增益只在配对口径成立**，差值口径在 0.5/2.0 mm 为负 ⇒ **种子对齐效应**，不重开；"
    "⑥ r(dCSI, dBIAS) = +0.4959／+0.5576／+0.6800 ⇒ 越涨分越过预报；"
    "⑦ **新口径纪律**：配对口径与差值口径必须并列报告。\n\n"
    "**边界**：事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0、S5 不换阈值；0.5/4.0 mm 无 bootstrap CI）"
    "⇒ **裁决不变：六臂判负、无一臂进十 seed、不投十 seed 算力**；"
    "**MASTER 不升版**（无数值与裁决变化）⇒ 新证据只进 `HANDOVER_E45.md` §4⑨、上层 MEMORY.md 状态段与三份逐日日志。\n\n"
    "**踩坑（本轮自写脚本连续 4 次报错，均为自身 bug）**：键名是 `A1_s0_ep00`（seed 段要 `lstrip(\"s\")`）；"
    "`\"%g\" % 2.0` 得到 `\"2\"` 而产物键是 `\"2.0\"` ⇒ 阈值必须用显式后缀表；"
    "THRS 改成二元组后所有解包点要同步改；S3 的 CI 键是 `ci_lo`/`ci_hi`/`crosses_zero` 而非 low/high；"
    "产物里 **POD/FAR 只记录在 thr=2.0**。⇒ **规则：写只读诊断脚本，产物键名必须先 dump 一次再写断言。**\n"
)


def insert_once(path, anchor, new_text, marker, label):
    if not path.exists():
        raise SystemExit("FILE MISSING: %s" % path)
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print("  already: %s [%s]" % (path.name, label))
        return
    if anchor not in text:
        raise SystemExit("ANCHOR MISS (%s) in %s" % (label, path))
    text = text.replace(anchor, new_text, 1)
    path.write_text(text, encoding="utf-8")
    print("  inserted: %s [%s]" % (path.name, label))


def append_once(path, marker, note):
    if not path.exists():
        raise SystemExit("FILE MISSING: %s" % path)
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print("  already: %s" % path.name)
        return
    path.write_text(text + note, encoding="utf-8")
    print("  appended: %s" % path.name)


def main():
    print("=== HANDOVER_E45.md  §4⑨ ===")
    insert_once(HANDOVER, HANDOVER_ANCHOR, HANDOVER_ITEM + "\n" + HANDOVER_ANCHOR,
                MARK + "新增证据", "handover")

    print("=== MEMORY.md 状态段 ===")
    insert_once(MEMORY, MEMORY_ANCHOR, MEMORY_NEW, "两种「中位数增益」口径必须并列", "memory")

    print("=== 逐日日志 ===")
    append_once(DAILY_UP, MARK, DAILY_NOTE)
    append_once(DAILY_PJ, MARK, DAILY_NOTE)
    append_once(LOG_1111, MARK, LOG1111_NOTE)

    print("=== 自动化 memory ===")
    append_once(AUTOMATION, MARK, AUTOMATION_NOTE)

    print("=== verify ===")
    for p, m in ((HANDOVER, MARK), (MEMORY, "两种「中位数增益」口径必须并列"),
                 (DAILY_UP, MARK), (DAILY_PJ, MARK), (LOG_1111, MARK), (AUTOMATION, MARK)):
        t = p.read_text(encoding="utf-8")
        n = t.count(m)
        assert n >= 1, "marker missing in %s" % p
        print("  %-22s marker_count=%d" % (p.name, n))
    print("OK")


if __name__ == "__main__":
    sys.exit(main())
