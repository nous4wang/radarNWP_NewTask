"""Round-43 log bookkeeping: correct one sentence, append the record.

What this corrects
------------------
The v2.48 daily note said the historical section 12 rows (v2.44-v2.47)
were deliberately left alone ("刻意不改写").  That was true when the
note was written, but it stopped being true within the same round: the
in-place branch-phrase fix (tools/_fix_v248_branch_phrase.py) DID
rewrite those rows, because the phrase they carried ("退化成常数") is
factually wrong, not merely outdated, and this project already rewrites
a changelog row in place when it carries a wrong conclusion -- the
v2.45 write-back did exactly that to the v2.44 row.

So the note is corrected rather than left to mislead, and the record of
what was done is appended to both daily logs.

Idempotent: running it twice changes nothing.
"""
from pathlib import Path
import sys

DAILY_UP = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-13.md")
DAILY_PJ = Path(".workbuddy/memory/2026-11-13.md")
LOG_1111 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")

STALE = (
    "**历史 §12 行（v2.44–v2.47）刻意不改写**（它们是审计痕迹），"
    "由新的 v2.48 行披露「『退化成常数』措辞已被取代」。"
)
FRESH = (
    "**历史 §12 行（v2.44–v2.47）按 v2.45 的先例就地改写**："
    "该行里的「退化成常数」不是「过时」而是**事实错误**（该格的 `scale_cv` "
    "3.0188 是六格最高，量通道根本没有变成常数）⇒ 按本项目既有做法"
    "（v2.45 就地改过 v2.44 行）用 `tools/_fix_v248_branch_phrase.py` 就地订正，"
    "并由新的 v2.48 行披露这次订正；**逐日日志仍按「只增不改」保留原样**。"
)

APPEND_2026_11_13 = (
    "\n\n- **（2026-11-13 17:0x 补记，第 43 轮）** E45 结案结论不变"
    "（六臂判负、不投十 seed 算力）；**新发现并已订正一处事实错误**："
    "`scale_is_constant` 是**析取**（`scale_cv < 0.05` **或** "
    "`scale_truth_corr < 0.3`），而结论正文一律写作「退化成常数」＝断言了 cv 分支；"
    "逐格重算证明 **cv 分支在 E45 全部六格从未触发**（最小 `scale_cv` 1.2358 ＝ "
    "底线的 24.7 倍），**只有相关性分支在 A4b 种子 2 触发一次**，而该格 `scale_cv` "
    "3.0188 恰是六格最高 ⇒ 实为**量级塌缩＋与真值脱钩**。**裁决零变化**"
    "（A4b 种子 2 的标志仍为真 ⇒ 预注册舍弃照样触发）。"
    "详见 MASTER §12 的 v2.48 行与 `HANDOVER_E45.md` §4⑧。\n"
)

APPEND_2026_11_11 = (
    "\n- **（2026-11-13 17:0x 补记，第 43 轮）** E45 结案结论不变"
    "（六臂判负、不投十 seed 算力）；**新发现并已订正一处事实错误**："
    "量通道的 `scale_is_constant` 是**析取**判据（cv<0.05 **或** corr<0.3），"
    "而结论正文把它一律写成「退化成常数」＝断言了 cv 分支；逐格重算证明 "
    "**cv 分支在 E45 全部六格从未触发**（最小 `scale_cv` 1.2358 ＝ 底线的 24.7 倍），"
    "**只有相关性分支在 A4b 种子 2 触发一次**，而该格 `scale_cv` 3.0188 恰是六格最高 "
    "⇒ 实为**量级塌缩＋与真值脱钩**（`scale_mean` 0.0316）。"
    "**A4b 仍按预注册舍弃**（该格的标志为真）⇒ **六臂判负、不投十 seed 算力不变**。"
    "详见 MASTER §12 的 v2.48 行。\n"
)


def patch(path, stale, fresh, label):
    text = path.read_text(encoding="utf-8")
    if stale in text:
        text = text.replace(stale, fresh)
        path.write_text(text, encoding="utf-8")
        print(f"  fixed stale sentence: {path} [{label}]")
    elif fresh in text:
        print(f"  already fresh: {path} [{label}]")
    else:
        raise SystemExit(f"ANCHOR MISS ({label}) in {path}")


def append_once(path, marker, note):
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"  already appended: {path}")
        return
    path.write_text(text + note, encoding="utf-8")
    print(f"  appended: {path}")


def main():
    print("=== correct the stale sentence in both daily logs ===")
    for p in (DAILY_UP, DAILY_PJ):
        if not p.exists():
            raise SystemExit(f"FILE MISSING: {p}")
        patch(p, STALE, FRESH, p.name)

    print("=== append the round-43 record ===")
    # both daily logs carry it (the project keeps the two mirrored)
    append_once(DAILY_UP, "补记，第 43 轮", APPEND_2026_11_13)
    append_once(DAILY_PJ, "补记，第 43 轮", APPEND_2026_11_13)
    append_once(LOG_1111, "第 43 轮", APPEND_2026_11_11)

    print("=== verify ===")
    for p in (DAILY_UP, DAILY_PJ, LOG_1111):
        t = p.read_text(encoding="utf-8")
        assert STALE not in t, f"stale sentence still in {p}"
        print(f"  {p.name}: stale=0, mentions v2.48={'v2.48' in t}, "
              f"mentions round 43={'43' in t}")


if __name__ == "__main__":
    sys.exit(main())
