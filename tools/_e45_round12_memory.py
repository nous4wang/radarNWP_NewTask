"""E45 第 12 轮独立复核的记忆写入（幂等 + 锚点唯一性自校验）。

写两处：
  1. 上层 MEMORY.md 的 E45 状态段末尾追加「第 12 次独立复核」注记
  2. 工作区当日日志 .workbuddy/memory/2026-11-12.md 追加本轮复核段

不碰 docs/MASTER.md、docs/experiments/*、scripts/、tests/ ——
裁决与写回均无变化，改它们只会制造无谓 diff。
"""
from pathlib import Path

MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
DAILY = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-12.md")

ANCHOR = "前十轮复核结论完全一致，无新增偏离。）**"
NOTE = (
    " **（2026-11-12 02:48 第 12 次独立复核：远端 273/291 全 `COMPLETED`、队列空、21 份评测齐、"
    "提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`、"
    "远端 `runs/e45/` 13 个 json 与本地 md5 **13/13 全同**；S3 六臂原始增益 CI 与配平后 CI **全跨零**"
    "（A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH`）；"
    "3-seed 中位数/过地板通过率（地板 0.0924、Oracle 0.1653）：B0 0.1218/2-3、A1 0.1184/2-3、"
    "A2 0.1120/2-3、A3a 0.1100/3-3、A3b 0.1161/3-3、A4 0.1439/2-3、A4b 0.1376/2-3；"
    "**新脚本 `tools/_e45_verify_round12.py` 不复用旧脚本函数、直接从三份原始探针 JSON 重取读数**："
    "门 A1 corr(mask,amp) +0.6103/+0.6145/+0.6881、A2 +0.4035/+0.3113/+0.3879 全 ≥0 且 "
    "`mask_at_peak` 0.72～1.0 全 >0.5 ⇒ 3/3 PASS；A3a `grad_top1pct_share` 0.5526/0.5666/0.4782、"
    "A3b 0.9712/0.9915/0.9775，`field_max_mm` 三 seed **全 402.4288 > 30** ⇒ 0/3 PASS；"
    "A4 `scale_cv` 2.0040/1.2358/1.7046、`scale_truth_corr` 0.7429/0.5521/0.6384、"
    "`scale_is_constant` 全 False ⇒ 3/3；**A4b 只 2/3**（s2 `scale_truth_corr` 0.2789 < 0.3 ⇒ "
    "退化成常数）；`maxratio_as_trained` 三 seed 全 `nan`；写回脚本连跑 **0 改动**、"
    "两个校验器 `VERDICT: PASS`、Gate 3 四处＋INDEX＋MEMORY 指针全部在位 ⇒ 结论无误，"
    "**未改 docs、未下载、未 sbatch**。前十一轮复核结论完全一致，无新增偏离。）**"
)

DAILY_NOTE = """
## 2026-11-12 02:48（本次运行 · 第 12 次独立复核）

**判定：远端 E45 全部完成且与前十一轮完全一致；裁决与写回均无需改动；本轮无远端改动（未下载、未 sbatch、未改 docs）。**

- 远端：提交器 **273** `COMPLETED`（05:04:04）、训练数组 275/283 与评测 274/282/290 全 `COMPLETED`、
  汇总 **291** `COMPLETED`（00:14:23）、**21 份评测齐**、**队列空**；提交器日志尾
  `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。
- **md5 13/13 全同**：远端 `runs/e45/` 13 个 json 与本地 `tools/_evals/e45/` 13 个 json 逐一相同
  ⇒ 无需重新下载。
- **S3 独立复核**（`tools/_e45_verify_round9b.py`）：六臂原始增益 CI **全跨零**、配平后 CI **全跨零**；
  A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH`
  ⇒ **六臂判负、不投十 seed 算力**成立。
- **3-seed 中位数 / 过地板通过率**（地板 0.0924、Oracle 0.1653）：B0 0.1218/2-3、A1 0.1184/2-3、
  A2 0.1120/2-3、A3a 0.1100/3-3、A3b 0.1161/3-3、A4 0.1439/2-3、A4b 0.1376/2-3
  ⇒ 高于对照中位数的只有 A4、A4b（BIAS 同时爆表 0.8362 / 2.8215）；**对照 B0 自己只 2-3 过地板**
  ⇒ 地板口径不稳，不构成晋级依据。
- **机制读数逐臂逐 seed 独立重算**（新脚本 `tools/_e45_verify_round12.py`，**不复用旧脚本函数、
  直接从三份原始探针 JSON 重取**，避免把旧脚本的口径 bug 继承进来）：
  - 门（A1/A2）：`corr(mask,amp)` A1 +0.6103/+0.6145/+0.6881、A2 +0.4035/+0.3113/+0.3879（全 ≥0），
    `mask_at_peak` 0.7224～1.0（全 >0.5）⇒ **两臂 3/3 PASS**；A2 干区比例升至 24.4 %（s0 14.4 %）。
  - 拓扑（A3a/A3b）：`grad_top1pct_share` A3a 0.5526/0.5666/0.4782、A3b 0.9712/0.9915/0.9775，
    `field_max_mm` 三 seed **全 402.4288 > 30** ⇒ **两臂 0/3 PASS，拓扑支按预注册结掉**。
  - 量通道（A4/A4b）：A4 `scale_cv` 2.0040/1.2358/1.7046、`scale_truth_corr` 0.7429/0.5521/0.6384、
    `scale_is_constant` 全 False ⇒ **3/3**；**A4b 只 2/3**（s2 `scale_truth_corr` 0.2789 < 0.3 ⇒
    退化成常数）；`mean_out_equals_T`：A4 s0/s2 True、s1 False，A4b 三 seed 全 False ⇒ 仍属实现缺陷。
  - `maxratio_as_trained`：A1/A2 三 seed **全 `nan`**（读数缺失，不是通过也不是失败）。
- **写回复核**：`_sync_v245.py` 连跑 **0 改动**（幂等，五个目标文件均 `unchanged`）、
  `_check_tables.py` **VERDICT: PASS**（`introduced=[]`）、`_check_v245_row.py` **VERDICT: PASS**
  （v2.44/v2.45 行均 4 cells，与 §12 表头一致）；MASTER 头部 **v2.45**、§4.47 卡片（L1147）、
  §9 E45 行（L1406）、§12 v2.44/v2.45 两行（L1992/L1993）、INDEX E45 行（L51）、
  上层 `MEMORY.md` 指针 v2.45 全部在位。
- 记忆：上层 `MEMORY.md` 状态段追加本轮复核注记；本文件追加本段。

### 本轮新增可复用经验

- **复核脚本要「不继承旧口径」**：第 10 轮刚修过 `tools/_e45_verify_round9b.py` 的三处口径 bug
  （对照基准误取 min 当中位数、过地板计数写成 `pass_rate*3>=2.9`、S3 的 CI 键误用 `ci_low/ci_high`）。
  若新复核脚本再 import 旧脚本的函数，**旧 bug 会静默传染**。本轮改为
  **从三份原始 `diag_probe_s{0,1,2}.json` 直接重取读数**（读数在 `arms/<臂>/readouts/` 下），
  输出与 MASTER v2.45 逐项一致 ⇒ 口径可独立验证，不靠「同一脚本自己证明自己」。
"""


def main():
    # --- MEMORY.md ---
    text = MEMORY.read_text(encoding="utf-8")
    if "第 12 次独立复核" in text:
        print("MEMORY already carries round-12 note -> skip")
    else:
        n = text.count(ANCHOR)
        if n != 1:
            raise SystemExit(f"ANCHOR MISS: expected 1 occurrence, found {n}")
        MEMORY.write_text(text.replace(ANCHOR, ANCHOR + NOTE), encoding="utf-8")
        print("patched:", MEMORY)

    # --- 当日日志 ---
    if DAILY.exists():
        text = DAILY.read_text(encoding="utf-8")
        if "第 12 次独立复核" in text:
            print("daily log already carries round-12 note -> skip")
        else:
            DAILY.write_text(text + DAILY_NOTE, encoding="utf-8")
            print("patched:", DAILY)
    else:
        DAILY.parent.mkdir(parents=True, exist_ok=True)
        DAILY.write_text(
            "# 2026-11-12 工作日志（radar-nwp-bench）\n" + DAILY_NOTE, encoding="utf-8"
        )
        print("created:", DAILY)


if __name__ == "__main__":
    main()
