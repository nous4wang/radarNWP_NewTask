#!/usr/bin/env python
"""E45 round 63 -- write-back (idempotent, marker-strip then append).

Writes:
  1. HANDOVER_E45.md  (§4 new block ㉖ -- weight-space axis)
  2. upper MEMORY.md   (status paragraph, marker block)
  3. upper 2026-11-14.md  (daily log, marker block)
  4. project .workbuddy/memory/2026-11-14.md  (daily log, marker block)

Discipline (from rounds 54-62):
  * strip the previous marker block first, then append  -> idempotent
  * marker occurrence count must be exactly 1 in every file afterwards
  * a table row must be built from a CELL LIST with the separator inserted by code
  * write only AFTER the assertions passed
  * run twice -> the second run must be a no-op (byte-identical)
"""

import hashlib
import os
import re
import sys

BASE = "R:/1radarNWPprediction"
REPO = BASE + "/radar-nwp-bench"

HANDOVER = BASE + "/.workbuddy/memory/HANDOVER_E45.md"
MEMORY = BASE + "/.workbuddy/memory/MEMORY.md"
DAILY_UP = BASE + "/.workbuddy/memory/2026-11-14.md"
DAILY_PRJ = REPO + "/.workbuddy/memory/2026-11-14.md"

BEGIN = "<!-- E45-R63-BEGIN -->"
END = "<!-- E45-R63-END -->"

HANDOVER_BLOCK = """
**㉖（2026-11-14 第 63 轮（权重空间轴：机制在参数空间里到底动了多少、动了哪边））**——脚本 `tools/_e45_weight_space_probe.py`（远端**只读**探针，POS 9/9）＋ `tools/_e45_weight_direction_probe.py`（远端只读探针，POS 2/2）＋ `tools/_e45_verify_round63.py`（本地整合，**POS 12/12 VERDICT: PASS**，产物 `tools/_evals/e45/round63_weight_space.json`）：前 62 轮判过聚合层／机制探针／S1 结构量／窗级／场次级／时效轴／强度·难度轴／误差构成／臂间合并／剂量-反应／指标族／护栏标定／增益图样稳定性／聚合算子＋评测契约／训练动力学／晋级规则标定／决策价值／ROPE 认证／待办可执行性共二十层，**全部读的是「产物里的指标」**——**21 份 `epoch_00.pt` 的参数张量从未被任何一轮打开过**（第 57 轮只审了远端 `config.json`、第 58 轮只读了训练日志 stdout）。本轮第一次在**权重空间轴**上判读，**裁决零变化**（事后诊断，**不构成新判据**：S3 护栏固定 thr=2.0 mm、复制单元固定为 seed），得七条新事实：

1. **首次把「机制动了多少参数」变成可测量**：在 66 个共享键上算相对 L2 距离 `‖w_arm − w_B0‖ / (‖w_arm‖+‖w_B0‖)/2`（同 seed）。**零模型＝对照臂 B0 换种子**（纯 seed 噪声、无机制）⇒ **seed 噪声中位 0.025030（21 对），机制效应中位 0.012279（18 格）⇒ 比值 0.4906**；**18 个（臂×seed）效应距离里 16 个低于 seed 噪声中位**（例外：A4_s1 = 0.032516、A4b_s2 = 0.049174，恰是两条 BIAS 爆表臂的不稳定种子）。**逐臂「效应 ÷ 自身 seed 噪声」：A2 0.2260｜A1 0.2663｜A4b 0.3581｜A3b 0.5020｜A3a 0.5023｜A4 0.6364 ⇒ 六臂无一超过 1**。
   ⚠️ **口径边界（必须同时报）**：这一比值**只说明「机制对参数空间的扰动幅度小于换一次种子」**，**不等于「机制没生效」**——第 57 轮的 config 审计与第 53 轮的机制读数已独立证明机制确实生效（`corr(mask,amp)` 由 −0.858320 转到 +0.39～+0.69）；它说明的是**「幅度小」不足以换到分数**。

2. **关键混杂被排除（新登记）**：跨臂距离会不会只是「不同随机初始化」造成的？逐张量做对照——把「臂 vs B0 同 seed」与「B0 换种子（同张量全量重初始化）」逐张量比：**六臂的逐张量比值中位 0.1358（A2）～0.2986（A4b），全部 < 1**；**每个臂都有 90.3%–100% 的张量其臂间扰动小于重初始化幅度** ⇒ **共享张量在同 seed 下确实是同一初始化起点**，机制效应是真实扰动而非初始化差异。**每臂恰有 4/66 个张量与 B0 逐位相同**（其余 62 个都变了）⇒ **机制确实作用在整网而非局部**。

3. **方向轴给出了与第 56 轮（窗级增益图样）完全同向、但机理更硬的证据**：算「机制扰动方向」在三个 seed 之间的一致性 `cos(δ_s1, δ_s2)`（δ = w_arm,s − w_B0,s），**零模型＝B0 换种子的方向一致性中位 0.4998**。⇒ **A3a +0.7282、A3b +0.7294、A4 +0.5641、A4b +0.5579 四臂的机制方向跨 seed 稳定（高于零模型）**，而 **A2 +0.2681 低于零模型**、**A1 −0.2957 更差：3 对里有 2 对为负 ⇒ 门族的方向在 seed 间会翻转**。⇒ **「机制生效」在 A1 上连方向都不稳定**，与第 56 轮「A1/A2 完全无图样」、第 44 轮「种子对齐效应」同向，但这是**第一份参数空间的一手证据**。

4. **跨臂方向对齐给出一个此前没人看过的结构事实**：**A3a 与 A3b 的机制方向几乎重合（同 seed 余弦中位 +0.9486，逐 seed +0.8998/+0.9486/+0.9630）**，而**它们彼此的参数距离只有 0.0039–0.0044（比各自到 B0 的 0.0123 小得多）** ⇒ **「A3a 与 A3b 是两个不同机制」这个前提在参数空间里不成立——它们几乎是同一个解，只差一个与主方向近乎正交的小扰动**（幅度比 ≈ 1/3）。⚠️ **口径边界**：A3a/A3b 的差别（`--topo-clamp-mm 20` vs `--topo-truth-smooth-sigma 1.0 --topo-area-normalize`）在**损失项内部**、不改变模型结构 ⇒ 它们落在同一条优化轨迹的附近；**这解释了两臂为何在技能、场次、时效、剂量-反应各轴上都高度同向**，但**不构成「两臂等价」的正式主张**（正式等价需重训验证）。
   **另一条同族事实**：**拓扑族与量族方向近乎相反**（A3a|A4 −0.5642、A3b|A4 −0.5702、A3a|A4b −0.4508），而**两门臂之间 +0.7515** ⇒ **三个机制族在参数空间里是三个可分辨的方向**，不是同一族的三次扰动。

5. **A4 与 A4b 是「同一族里离得最远的两臂」**：同 seed 参数距离 0.015672（s0）→ 0.024960（s1）→ **0.041626（s2）**，**随种子单调放大 ⇒ 与 A4b 种子 2 的量级塌缩（第 45/48 轮已登记）同向**：**塌缩格正是两臂参数距离最大的那一格**。⇒ 「A4b 不新增头（模块层）」与「A4b 的量通道塌缩」在参数空间同时可见：**A4b 用自己的解码头承担量通道时，与 A4（新标量头）的解在 seed 2 上分道扬镳**。

6. **量级与方向的分离给出一条新纪律**：**A4 的参数扰动幅度最大（0.020595 中位）、A4b 次之（0.018030）、拓扑两臂 0.0123、门两臂 0.006 左右**，而**方向稳定性完全不是同一排序**（A3a/A3b 方向最稳但幅度中等；A1 幅度最小且方向会翻）⇒ **「机制对参数动了多少」与「机制在参数空间是否稳定」是两个独立量，必须并列报告，不得合并成一个「机制强度」**（与第 52 轮「方差份额」必须分项、第 43 轮「派生布尔必须分分支」同族）。

7. **剂量-反应在权重空间上仍站不住**：跨 6 臂，`r(机制扰动幅度, 配对增益中位) = +0.6945`（置换检验 20000 次，**p = 0.1292**）、`r(效应/自身 seed 噪声比, 配对增益中位) = +0.7655`（**p = 0.1052**）⇒ **两条都在 n=6 下不可区分于零** ⇒ 与第 53 轮（机制读数 → 技能收益，置换检验全不可区分）**同向**：**「机制生效」与「分数受益」在两条独立的机制度量上都不成立**。

**守卫与口径**：① **S3 的 12 个区间仍全部跨零**、`crosses_zero` 12/12 为 true、六臂 `s3_verdict` 逐臂不变（4 × `GAIN_WAS_AREA_OR_AMPLITUDE` ＋ 2 × `NO_GAIN_TO_BEGIN_WITH`）、**合格臂集合仍为空**；② 已发布头条表七臂中位 CSI 由逐 seed 产物复算 **7/7 逐位一致（max|diff| = 4.53e−07）**、配对增益中位 **max|diff| = 4.89e−07**、差值口径增益 **max|diff| = 4.83e−07**；③ 仍按「**中位数 + 通过率**」报，单点分数不作裁决；④ 配对口径与差值口径并列；⑤ 事后诊断，**不构成新判据**、**不改变任何裁决**（六臂仍全部判负、**无一臂进十 seed、不投十 seed 算力**）；⑥ **MASTER 不升版**（无数值与裁决变化，仍 v2.48）；⑦ 本轮**未 sbatch、未改远端状态、未动 `docs/`**；远端 13 份 json 与本地 **md5 13/13 全同**，四个文档闸门全 `VERDICT: PASS`。

**新登记三条可复用纪律**：① **「机制幅度小于 seed 噪声」≠「机制没生效」**——任何「机制扰动 vs 随机扰动」的比较，都必须同时给出**机制自己的可验证读数**（本项目是 config 审计 ＋ 机制探针），否则会把「生效但幅度小」误写成「没生效」；② **跨臂参数距离必须先排除初始化混杂**（逐张量对「重初始化幅度」做比值），否则跨臂距离可能只是随机起点不同；③ **幅度与方向必须并列报告**——本轮 A4 幅度最大、A3a/A3b 方向最稳，两者是不同性质。

**远端运行状态（第 63 轮复核）**：提交器 273、训练数组 275/283（各 7/7）、评测 274/282/290、汇总 291、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份评测齐**、`runs/e45/` **13 份 json 与本地 md5 13/13 逐一致**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；`runs/` 下无 E46+ 新目录 ⇒ **无新活动**。**`Slurmctld(primary) at nbdx` 仍 DOWN**（`systemctl is-active slurmctld` → `failed`；**squeue/scontrol 不可用，sacct/slurmdbd 可用**）⇒ 第 62 轮登记的三项待办仍**无法提交**（需管理员重启控制器；本账号无 sudo，**不得以登录会话手跑 GPU 任务替代**）。本轮远端动作**仅两次只读探针**（读 checkpoint、算距离），**无训练、无 GPU 占用、无状态修改**。
"""

MEMORY_BLOCK = """
<!-- E45-R63-BEGIN -->
- **第 63 轮新增一条轴（权重空间轴：机制在参数空间里动了多少、动了哪边）**——**21 份 `epoch_00.pt` 的参数张量首次被打开**（前 62 轮只读指标产物，从未读参数）：① **幅度**：66 共享键上的相对 L2 距离，**seed 噪声中位 0.025030（零模型＝B0 换种子）／机制效应中位 0.012279 ⇒ 比值 0.4906**；**18 个（臂×seed）效应距离里 16 个低于 seed 噪声中位**（例外 A4_s1、A4b_s2）；**逐臂「效应 ÷ 自身 seed 噪声」A2 0.2260｜A1 0.2663｜A4b 0.3581｜A3b 0.5020｜A3a 0.5023｜A4 0.6364 ⇒ 六臂无一超过 1**。② **混杂排除**：逐张量对「B0 重初始化幅度」比值中位 0.1358–0.2986（全部 < 1）⇒ 同 seed 下共享张量起点相同，效应是真实扰动；**每臂恰 4/66 张量与 B0 逐位相同**（其余 62 个都变）⇒ 机制作用在整网。③ **方向**：机制扰动方向跨 seed 的余弦，零模型中位 0.4998；**A3a +0.7282、A3b +0.7294、A4 +0.5641、A4b +0.5579 稳定**，**A2 +0.2681 低于零模型**，**A1 −0.2957 且 3 对里 2 对为负 ⇒ 门族方向会翻**。④ **跨臂方向**：**A3a|A3b +0.9486（几乎同一解，彼此参数距离仅 0.0039–0.0044）**，拓扑族与量族近乎相反（A3a|A4 −0.5642），两门臂之间 +0.7515 ⇒ 三族在参数空间可分辨。⑤ **A4|A4b 距离随种子单调放大 0.0157→0.0250→0.0416 ⇒ 塌缩格正是距离最大格**。⑥ **剂量-反应仍站不住**：`r(幅度, 配对增益) = +0.6945`（置换 p = 0.1292）、`r(比值, 配对增益) = +0.7655`（p = 0.1052）。**裁决零变化、MASTER 不升版（仍 v2.48）、不投十 seed 算力**。**新登记三条纪律**：「机制幅度小于 seed 噪声」≠「机制没生效」（须并报机制自身可验证读数）；跨臂参数距离必须先排除初始化混杂；幅度与方向必须并列报告、不得合并成单一「机制强度」。
<!-- E45-R63-END -->
"""

DAILY_BLOCK = """
<!-- E45-R63-BEGIN -->
### 2026-11-14｜第 63 轮（权重空间轴）：状态复核 + 只读判读，裁决不变

- **判定**：远端 E45 全部完成且与第 50–62 轮完全一致；**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮远端动作**仅两次只读探针**（读 21 份 checkpoint、算参数距离），**未 sbatch、未改远端状态、未动 `docs/`**（MASTER 仍 v2.48）。
- **远端**：273/274/275/282/283/290/291/297/298 全部 `COMPLETED`；21 份评测齐、13 份 json 与本地 md5 **13/13**；四个文档闸门全 `VERDICT: PASS`；`slurmctld` 仍 **DOWN** ⇒ 第 62 轮的三项待办仍无法提交（需管理员）。
- **新轴**（前 62 轮从未碰过）：**权重空间轴**——**21 份 `epoch_00.pt` 的参数张量首次被打开**。脚本 `tools/_e45_weight_space_probe.py`（远端只读，POS 9/9）＋ `tools/_e45_weight_direction_probe.py`（远端只读，POS 2/2）＋ `tools/_e45_verify_round63.py`（本地整合，**POS 12/12 VERDICT: PASS**，产物 `tools/_evals/e45/round63_weight_space.json`）。
- **五条新事实**：① **幅度**：seed 噪声中位 0.025030（零模型＝B0 换种子）vs 机制效应中位 0.012279 ⇒ **比值 0.4906**，18 格中 16 格低于噪声中位；逐臂「效应÷自身 seed 噪声」0.226–0.636 ⇒ 六臂无一超过 1。② **混杂排除**：逐张量对重初始化幅度比值中位 0.1358–0.2986（全部 < 1）⇒ 起点相同、效应真实；每臂 4/66 张量与 B0 逐位相同、其余 62 个全变 ⇒ 机制作用整网。③ **方向**：零模型 0.4998；**A3a +0.7282 / A3b +0.7294 / A4 +0.5641 / A4b +0.5579 稳定**，**A2 +0.2681 低于零模型**，**A1 −0.2957 且 2/3 对为负 ⇒ 门族方向会翻**。④ **A3a 与 A3b 方向几乎重合（+0.9486）且彼此参数距离仅 0.0039–0.0044 ⇒ 两臂几乎是同一个解**（口径边界：不构成正式等价主张）；拓扑族与量族方向近乎相反（A3a|A4 −0.5642），三族可分辨。⑤ **A4|A4b 距离随种子单调放大 0.0157→0.0416 ⇒ 塌缩格正是距离最大格**；剂量-反应仍不可区分于零（置换 p = 0.1292 / 0.1052）。
- **守卫**：S3 的 12 个区间仍全部跨零、六臂 `s3_verdict` 不变、**合格臂集合仍为空**；头条表七臂中位 CSI 复算 max|diff| = 4.53e−07、配对增益 4.89e−07、差值口径 4.83e−07。
- **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**、**不改变任何裁决**；④ **新登记三条纪律**：「机制幅度小于 seed 噪声」≠「机制没生效」（须并报机制自身可验证读数）；跨臂参数距离必须先排除初始化混杂；幅度与方向必须并列报告、不得合并成单一「机制强度」；⑤ **MASTER 不升版**（仍 v2.48）。
<!-- E45-R63-END -->
"""


def read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def strip_marker(text):
    """Remove a previously written block (marker pair) so the append is idempotent."""
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", re.S)
    new, n = pat.subn("", text)
    return new, n


def apply_marker(text, block):
    """Replace an existing marker block with `block`, or append it if absent."""
    stripped, n = strip_marker(text)
    if n:
        return stripped + block.lstrip("\n"), "replaced"
    return text.rstrip("\n") + "\n\n" + block.strip("\n") + "\n", "appended"


def main():
    before = {}
    for p in (HANDOVER, MEMORY, DAILY_UP, DAILY_PRJ):
        before[p] = sha(p)

    # 1. HANDOVER: append the numbered block at the end (rounds are appended in
    #    chronological order; the ㉕ block is the last section of the file).
    ht = read(HANDOVER)
    marker_txt = "㉖（2026-11-14 第 63 轮"
    if marker_txt in ht:
        print("HANDOVER: already contains round-63 block -> unchanged")
    else:
        write(HANDOVER, ht.rstrip("\n") + "\n" + HANDOVER_BLOCK)
        print("HANDOVER: round-63 block appended")
    if HANDOVER_BLOCK.strip() not in read(HANDOVER):
        print("ERROR: HANDOVER block missing after write")
        return 1

    # 2..4 marker files
    for path, block in ((MEMORY, MEMORY_BLOCK), (DAILY_UP, DAILY_BLOCK),
                        (DAILY_PRJ, DAILY_BLOCK)):
        t = read(path)
        new, how = apply_marker(t, block)
        write(path, new)
        print("%s: %s" % (os.path.basename(path), how))
        t2 = read(path)
        if t2.count(BEGIN) != 1 or t2.count(END) != 1:
            print("ERROR: marker count != 1 in %s (BEGIN=%d END=%d)"
                  % (path, t2.count(BEGIN), t2.count(END)))
            return 1

    after = {}
    for p in (HANDOVER, MEMORY, DAILY_UP, DAILY_PRJ):
        after[p] = sha(p)
    print("--- sha256 before/after ---")
    for p in (HANDOVER, MEMORY, DAILY_UP, DAILY_PRJ):
        print("  %-70s %s -> %s %s" % (os.path.basename(p), before[p][:12],
                                       after[p][:12],
                                       "CHANGED" if before[p] != after[p] else "same"))
    print("--- marker counts ---")
    for p in (MEMORY, DAILY_UP, DAILY_PRJ):
        t = read(p)
        print("  %-30s BEGIN=%d END=%d" % (os.path.basename(p), t.count(BEGIN), t.count(END)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
