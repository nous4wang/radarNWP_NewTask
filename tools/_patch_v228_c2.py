"""C2 closeout patch for v2.28 / E38.

After _sync_v228.py, the literal C2 condition ("no 'awaiting user adjudication'
wording inside MASTER §9 and §4.0") was not yet met: three phrasing instances
remained in the §4.0 table, and several historical cards lacked an explicit
history marker. This patch rewrites wording (meaning unchanged), adds history
cross-references, and normalises E38 to CRLF.

Zero compute: Markdown only. No training / inference / remote.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "docs/MASTER.md"
INDEX = ROOT / "docs/experiments/INDEX.md"
E38 = ROOT / "docs/experiments/E38-open-items-final-adjudication.md"


def load(p: Path) -> str:
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def save_crlf(p: Path, text: str) -> None:
    p.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))


PATCHES = [
    ("4.0-E20-row", MASTER,
     "按机制审查而非按名词立项；只有分位数重标定和蒸发门控值得按干净对照实测，其余须用户裁决或否决。 | `docs/experiments/E20-five-directions-gate-review.md` |",
     "按机制审查而非按名词立项；只有分位数重标定和蒸发门控值得按干净对照实测，其余当时未立项（**其中 D1 / D5 已于 v2.28 / E38 结案舍弃**）。 | `docs/experiments/E20-five-directions-gate-review.md` |"),

    ("4.0-E35-row", MASTER,
     "§15 三幕重写、README、3g / 3j / 3l 与 E36 正式实验仍待用户裁决（**该句是 v2.26 的历史状态**：§15 已于 v2.27 重写、README 状态声明已于 v2.28 更新、3g / 3j / 3l / E36 已于 v2.28 / E38 结案）。",
     "§15 三幕重写、README、3g / 3j / 3l 与 E36 正式实验当时留待后续（**该句是 v2.26 的历史状态**：§15 已于 v2.27 重写、README 状态声明已于 v2.28 更新、3g / 3j / 3l / E36 已于 v2.28 / E38 结案）。"),

    ("4.0-E38-row", MASTER,
     "| §4.40 | 把 §9 最后五项「待用户裁决」一次性结案，再让派生层抄权威事实 |",
     "| §4.40 | 把 §9 最后五项悬空裁决一次性结案，再让派生层抄权威事实 |"),

    ("4.20-card", MASTER,
     "**裁决**：按机制审查而非按名词立项；只有分位数重标定和蒸发门控值得按干净对照实测，其余须用户裁决或否决。",
     "**裁决**：按机制审查而非按名词立项；只有分位数重标定和蒸发门控值得按干净对照实测，其余当时未立项（**该句是 v2.10 的历史状态**：D1 / D5 已于 v2.28 / E38 结案舍弃）。"),

    ("4.37-card", MASTER,
     "§15 三幕重写、README 更新、3g / 3j / 3l 与 E36 正式实验**仍待用户裁决**，本轮不擅自关闭也不擅自执行。",
     "§15 三幕重写、README 更新、3g / 3j / 3l 与 E36 正式实验**当时仍留待裁决**，本轮不擅自关闭也不擅自执行（**该句是 v2.26 的历史状态**：§15 已于 v2.27 重写、README 状态声明已于 v2.28 更新、3g / 3j / 3l / E36 已于 v2.28 / E38 结案）。"),

    ("12-v228-history-list", MASTER,
     "§12 v2.26 行与 §4.37 卡片里的同类表述保留为**历史描述**并加交叉注。",
     "§12 v2.10 / v2.26 行与 §4.20 / §4.37 卡片里的同类表述保留为**历史描述**并加交叉注；§4.40 卡片与 §10.2 / §9 舍弃清单里的同类字样是「不可再写」的否定式引用，不是状态声明。"),

    ("index-E38-row", INDEX,
     "| E38 | §4.40 | 把 §9 最后五项「待用户裁决」一次性结案，再让派生层抄权威事实 |",
     "| E38 | §4.40 | 把 §9 最后五项悬空裁决一次性结案，再让派生层抄权威事实 |"),

    ("E38-c2-table", E38,
     "| §12 v2.26 行、`§4.37` 卡片末尾 | 「…仍待用户裁决」 | **保留原文**（历史记录，注明「该描述已于 v2.28 结案，见 E38」） |",
     "| §12 v2.10 / v2.26 行、§4.20 / §4.37 卡片正文 | 「…须用户裁决或否决」/「…仍待用户裁决」 | **保留原文 + 加历史交叉注**（注明「该句是 v2.10 / v2.26 的历史状态，已于 v2.28 / E38 结案」） |\n| §4.0 的 §4.20 / §4.37 / §4.40 三行 | 表格裁决列与描述列里的同类字样 | **C2 收口补丁**（`tools/_patch_v228_c2.py`）改写为等义、不含该字样的表述；语义不变，§4.40 卡片正文（§4 正文，非 §4.0 表格）与 §10.2 / §9 舍弃清单里的否定式引用保留 |\n| 全文 grep 命中项 | §4.40 卡片、§10.2「不可再写」、§12 v2.28 口径说明 | 属描述性或否定式引用，**不是状态声明**；§6 复现命令已注明需人工区分 |"),

    ("E38-repro-comment", E38,
     'grep -n "待用户裁决\\|须用户裁决\\|待裁决准入" docs/MASTER.md   # 仅剩历史叙述',
     'grep -n "待用户裁决\\|须用户裁决\\|待裁决准入" docs/MASTER.md   # 命中项应全为历史叙述或「不可再写」的否定式引用，需人工区分'),
]


def main() -> int:
    cache = {}
    for label, path, old, new in PATCHES:
        text = cache.get(path)
        if text is None:
            text = load(path)
        count = text.count(old)
        if count != 1:
            print(f"ABORT [{label}] {path.name}: expected exactly 1 match, found {count}")
            return 1
        cache[path] = text.replace(old, new, 1)
        print(f"  ok [{label}] {path.name}")

    for path, text in cache.items():
        save_crlf(path, text)
    # E38 must use the docs' CRLF convention even if no patch touched it.
    save_crlf(E38, load(E38))

    # C2 verification: literal wording inside §4.0 and §9 must be zero.
    master = load(MASTER).splitlines()
    pat = re.compile(r"待用户裁决|须用户裁决|待裁决准入")
    spans = {}
    start_40 = next(i for i, l in enumerate(master) if l.startswith("### 4.0 "))
    end_40 = next(i for i, l in enumerate(master) if l.startswith("### 逐次尝试"))
    start_9 = next(i for i, l in enumerate(master) if l.startswith("## 9. "))
    end_9 = next(i for i, l in enumerate(master) if l.startswith("## 10. "))
    spans["§4.0"] = master[start_40:end_40]
    spans["§9"] = master[start_9:end_9]
    bad = 0
    for name, lines in spans.items():
        hits = [(i + 1, l[:80]) for i, l in enumerate(lines) if pat.search(l)]
        print(f"C2 {name}: {len(hits)} hit(s)")
        for i, l in hits:
            print(f"    {name} line {i}: {l}")
        bad += len(hits)
    if bad:
        print(f"PATCH: FAIL ({bad} literal hits inside §4.0 / §9)")
        return 1
    print(f"PATCH: PASS ({len(PATCHES)} replacements, C2 clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
