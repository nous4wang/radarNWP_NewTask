"""Consolidate the project's long-term memory.

The memory had grown past what can be injected, and it was truncated on
the way in -- which means the next session reads it in pieces. This
rewrites it around what the memory is actually for: the rules that stop
a mistake from being made twice, and the numbers needed to read a result
correctly. Everything else lives in the master document, which is
pointed at from the top.

The consolidation keeps:
  - the environment and the rule about where heavy work runs,
  - the data gotchas, each with the wrong way it was done first,
  - the code conventions that are not obvious from the code,
  - the findings that change how a number has to be read,
  - the protocol and the pointer to the master document.

It drops the duplicates: several findings now appear in both the memory
and the master document, and the master document is the one that is
versioned with the code.
"""
import pathlib

MEMORY = pathlib.Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")

TEXT = """# radar-nwp-bench — 项目长期记忆

> **权威文档是 `docs/MASTER.md`**（v2.0，900+ 行）。本记忆只保留**规则**与
> **承重数字**；详细结果、代码骨架、实验规程一律查主控文档。
> 任何任务开始前先读主控文档（§3 细节处理、§4 已做尝试、§9 待办、§10 结论分层）。

## 执行环境（硬性要求）

**训练、评测、打包等重活一律在远程跑，本地不跑。**

| | 位置 |
|---|---|
| 代码 | `/home/wf/radarNWP`（远端） |
| 数据 | `/home/wf/data/radarNWP/BJ`（原始）/ `BJ_mmap`（mmap，训练用） |
| 环境 | `/home/wf/.conda/envs/radar_nwp`（torch 2.5.1+CUDA，8×RTX 4090） |
| 远程 | `183.136.129.105:1003` 账号 `wf`（密码见会话），助手 `tools/remote.py` |
| 本地 | `R:/1radarNWPprediction/radar-nwp-bench` —— **只做代码编辑 + 单测** |

服务器自带 `environment.yaml` 为权威版本，上传勿覆盖。本地可跑：
`tests/run_tests.py`（546/546，离线秒级）、`scripts/smoke_test.py`（55 项）。
**任何 train/evaluate/pack_data 不在本地执行。**

## 数据与几何（勿再踩；每条都记了先犯的错）

1. 两网格：RADAR/LABEL 301×301@0.01°，NWP 167×167@0.03°。NWP 范围**更大**（5.01° vs 3.01°）。
   配准 = ×3 双线性 → 501×501 → 中心裁剪 → 301×301（`coordinates.py`）。
   ❌ 曾对两者用相同 `[y0:y0+sz, x0:x0+sz]` 裁剪 → 每个 NWP 像素都空间错位。
2. NWP 时间戳 = 有效时刻；`forecast` 策略 = 分析小时 + 覆盖 T+60 的预报小时（2 片）。
   ❌ 曾只在过去查 NWP → 10 帧雷达窗跨 54 min，每步解析到同一整点 → NWP 成静态重复。
3. **评测必须拼全场 301×301**（`metrics/stitching.py`）。中心 crop 仅 2.6 % 降雨、
   中位 3.8 %；逐 tile 打分也错（重复计分、风暴切半）。谱分析同样必须全场。
4. FORMAT 216 文件 = 全国 216 个独立风暴窗口；BJ 的 62 是子集；case↔索引映射未发布。
5. **断帧 case 23 个**（判据 = 雷达序列内 6 min 跳步）：`IRREGULAR_CASES`@`data/index.py`，
   `split_cases()` 内扣除。洁净划分 23/8/8，断帧划分 23，零重叠。
   ❌ 曾放 4 个带五位后缀的不存在 ID → 洁净划分混入全部缺帧 case。
6. 窗口按**时间戳**建不能按位置（否则断帧 case 目标帧落 T+12）；用
   `index.add_minutes(issue_time, ±k*6)`。❌ 曾自己写 `(stamp//100)*60+(stamp%100)` →
   把 `MMDD*10000+HHMM` 读成跨午夜（4367 步里 20 步算错、漏检 1 个空洞）。
   插值要重建**真正缺失的那一帧**（时间上居中的），不是覆盖好帧。
7. 格式：RA 是 **6 分钟累积**（10 帧 = 1 h），`clean_label_rain` = `raw×0.1` → mm/6min。
   **档案里没有任何 NWP 累计降水变量** —— NWP 给的是大气状态，不是它产生的降水。
   唯一 mm 单位是 `PWAT`（可降水量，mean 23.99）→ 给降水**上界**。
   → 管线不需要 10 帧累加；论文里阈值必须写 "mm per 6 min"。
8. **RADAR 与 LABEL 时间戳 98/98 对齐** → 历史 11 帧的地面降水真值 `y_hist` 逐帧可得。
   同帧 CR 与 LABEL 逐像素相关仅 **r = +0.17**（跨模态鸿沟：dBZ ∝ ΣD⁶ vs 6 min 通量 ∝ ΣD³v(D)，
   中间隔着雨滴谱/云下蒸发/雨幡/风暴移动）→ "雷达反演雨量"本身是病态问题。
9. NWP 逐变量归一化（跨 6 个数量级）。FORMAT 列 36 个变量，**BJ 仅 27 个**
   → CIN/DCAPE/MLCIN/SBCIN 不可用，"能量释放"叙事不可验证。
10. **norm-v2**：旧常数把散度场 std 高估 **2.7–4.6 倍**，而散度正是量子插件读的
    动力学信号 → 插件当时只读到约 1/3。版本戳 `NWP_PIPELINE_VERSION` 使旧 ckpt 失效。
11. 标签：`radar_log_transform` = log1p(dBZ)/6（dataset 无条件调用）。Z-R 须先
    `expm1(x*6)` 反归一化。**CSI 全 0 / FAR=NaN 先查代码再怪模型。**
12. BCE logit 不能直接用 log1p(mm)（恒 ≥0 → sigmoid ≥0.5 → 干燥像素无法表达"无雨"）。
    正解 logit = log1p(mm) − log1p(阈值)。
13. 输出头勿硬 `nn.ReLU()` 收尾（初始几乎全在阈值下 → 零梯度 → 常数场）。
    正解：卷积收尾 + `floor_at_zero`。
14. 光流 `grid_sample` 在 `base+d` 采样会把内容移到 `x-d`，外推须 `base − u·t`；
    像素→网格系数 **2/(W−1)** 而非 0.05。
15. 插件投影须保空间结构：真实 NWP 场相邻像素相关 0.9996，GroupNorm(1) 投影后 0.9907，
    InstanceNorm2d 只 0.8662。

## 代码约定

16. 注册表开放 `@register_model`，基线在顶层 `models/<name>/model.py`（**按路径加载**，
    模块名 `_rnb_model_<name>` → **跨包 import 会重复注册**，共享骨架要放 `common.py`）。
17. `configs/` 真被读取（命令行 > 配置文件 > 脚本默认）。`--model-kwargs` 是 **JSON 对象**，
    不是 `key=value`。
18. **模型接缝固定为 `model(x_radar, x_nwp)`**。要额外输入（如 R2 的地面实况）
    只能**挂在模型属性上**（`obs_y_hist`），不能加第三个位置参数
    —— `evaluate.py` 统一调两参，加了第三个就评不了。
19. `x_nwp=None` 在接缝解析（`BasePrecipitationModel.__call__` 统一置零）；
    要判断"NWP 是否真实存在"必须在 resolve **之前**。
20. NWP 消融两套：`--no-nwp`（训练期不给）/ `--zero-nwp`（推理期置零）。
21. `is_learnable=False` 的解析基线不训练（pysteps_adv）。
22. 评测阈值可配 `--thresholds`；**不传时必须退回 `DEFAULT_THRESHOLDS`**
    （曾传 None → 验证器迭代 None → 任何不带阈值的评测在第一个样本就崩）。

## 已确立的结论（数字为准，细节查主控文档 §4/§10）

23. **强降水塌陷是标准损失的性质，不是方法族的性质**：18 模型矩阵，CSI@2.0 中位数
    0.0035 → 0.0262（逐模型倍数中位 **5.5×**，18/18 全部抬升）；CSI@4.0 中位
    0.0002 → 0.0078（44.6×）。❌"深度方法欠预测强降水是结构性缺陷"。
24. **⚠️ 强降水的地板是持久性外推，不是 0**：把最后一帧原样复制（Z-R 转雨量），
    同一 dataset/verifier/缝合，n=538 → **CSI@2.0 = 0.0924、CSI@0.5 = 0.2102**。
    pysteps_adv 0.101 **只比它高 9 %**；而 18 模型最好的 mmf_rnn 0.0896 **仍低于它**。
    → **主表必须加持久性这一行**；任何新方法的 CSI@2.0 须 > 0.0924 才算有信号。
25. **⚠️ 早停峰值在 epoch 2，不是 16**：`runs/epoch_curve/`（21 份逐 epoch 评测）
    ep0/1/2 = 0.129/0.138/**0.153**，ep19 = 0.027（**峰值是末轮 38.9 倍**），
    CSI@4.0 峰值 0.0884 是末轮 629 倍。旧文档的 "ep16 = 0.1183" 来自**半训练**的 D1 run。
    ep1 的 **BIAS = 0.973（≈持久性）** → 早期模型实质在做 QPE+持久性。
26. **⚠️ score-aware 的运行方差达 1.4 倍**（同配置同 seed 两次 ep19 = 0.0384 / 0.027），
    而标准损失两次几乎一致（0.0040 / 0.0039）→ **Δ 小于 1.4 倍不能称为提升**；
    任何单次运行必须配多 seed + bootstrap CI。
27. **CSI@2.0 恒为 0 的三个构造性缺陷**（`docs/THREE_DEFECTS.md`）：
    ①档位代表值取下界=验证阈值→重构值卡在 1.999797 mm（修法 `level_mode="midpoint"`，
    顶档 @4.0 命中 795→964 446，**1213 倍**）；②tier_weights 按位置索引（按阈值查表）；
    ③概率通道按位置匹配阈值（按阈值查通道 + 公布 `channel_thresholds`）。
    **教训：分数恰好为 0 时先怀疑构造，再怀疑模型；只看分布（分位数/不同取值数）才能发现。**
28. **量子插件是确定的负面结果**（干净对照，全矩阵复现）：score-aware 下插件
    CSI@2.0 = 0.0122 vs 骨干 0.0384（比值 **0.32**，比标准损失下的 0.50 更差）。
    机制：倾向幅值 vs CAPE 消耗 Pearson r 应为负却是 +0.191/+0.263；高通 64 px 后
    降到 **+0.021**（相关全由大尺度携带）；倾向幅值 vs 普通投影幅值相关 **0.869**
    （只是投影的单调变换）。**论文不能写"插件更优/捕捉能量释放"。**
29. **NWP 的用法比 NWP 的物理量本身更重要**：M1 concat vs radar_only CSI@2.0 持平
    但 CSI@0.1 **+0.064**；cross-attention −48 %；delta 持平；early concat −35 %；
    quantum −68 %。**最朴素的接入最好，越复杂越差。**
30. **"损失是瓶颈、架构无关"不成立**：Spearman ρ(std@2.0, sa@2.0) = **0.19**；
    但新排序与 std@0.5 高相关 ρ = **0.75**、R² = 0.53；离散度 sa **7.3×**。
    → **目标函数是强降水得分的必要非充分条件**；只是把既有技巧平移到强降水档。
31. **pysteps_adv 未被超过**（0/18 超过其 CSI@0.5/2.0）。CSI@0.1 下降但没跌破 0.1
    （最小 0.1963）→ "score-aware 站不住"不成立。FAR@0.5 的代价不推广（均值持平）。
32. **robustness 与 clean 划分不可直接比**（case 不相交、降水强度不同）。容错卖点需
    `--no-interpolate` **配对**实验。
33. 强降水阈值用 2 mm / 4 mm（16 mm 仅 0.0016 % 像素、噪声主导）。
34. **数据层加 `y_hist` 没有改变既有基线**（基线零实验 VERDICT: PASS）：新代码下重评
    既有 checkpoint，simvp_v2 的 CSI@0.1/0.5/2.0/4.0 = 0.4567/0.1437/0.0384/0.0078
    与矩阵**逐位一致**，n=538 一致，**命中数完全相同**；M1 concat 同样一致。
    → 旧矩阵基线仍可用，不必全基线重跑。

## 研究协议（强制）

**四道关卡**（主控文档 §11）：

| 关卡 | 动作 | 关键点 |
|---|---|---|
| Gate 0 | 读 MASTER.md | §3 细节处理、§4 已做尝试、§9 待办、§10 结论分层 |
| Gate 1 | **排查** | 试过吗？结论？**按机制比对，不按名词**；缺什么条件？缺就写待办，不硬上 |
| Gate 2 | 测试 | 判据**先写后跑**；对照只差一个变量；训练只在远端；出数前过**数据可信性七项** |
| Gate 3 | **纳入** | 写回 §4（动机→实现→结果→结论→状态）、更新 §9、§10、§12 变更日志 |

**Gate 1 特别重要**：本项目已两次踩"换个说法是同一个东西"的坑
（逐像素非对称 BCE ≡ 分层损失，两次都失败）。§10.2 列了**不可再写**的说法。

**载体**：技能 `research-master-doc`（`~/.workbuddy/skills/research-master-doc/`，
含 MASTER 模板 + `check_master.py`）；已合并进 aihero 的 `research` 技能；
项目测试 `tests/test_master_document.py`（文档结构被回归验证）。

## 远端操作坑（血泪）

- **扫矩阵脚本是"单趟遍历"**：遇"卡都忙"即 `continue`，跳过的模型不再回来 → 派发须排队。
- **train.py 每 epoch 都写 last.pt** → 半训练模型也有 ckpt。判据 = 日志有 `epoch 19:` 行；
  **不得用 `ls -dt .../*/last.pt` 选 ckpt**（会选到正在训练的那份）。→ 已犯错：D1/D2。
- **远端脚本可能被执行多次**：两个实例写同一输出目录。排查 `ps -eo pid,pgid,etime,args |
  grep '[r]un_xxx.sh'`，杀旧进程组 `kill -- -<pgid>`。
- **SSH 断开后台任务会被杀** → 用 `tools/remote.py --script X --detached LOG`
  （paramiko 起独立会话；`run()` 走 PTY 的方式不行）。
- **Git Bash 会把 `/tmp/...` 转成 Windows 路径** → 远端命令加 `MSYS_NO_PATHCONV=1`。
- 禁 `pkill -f` 停训练；用 `ps -eo pid,args | grep -- "--model xxx " | awk '{print $1}'`。
- 共用 GPU 须显式分配；`nvidia-smi --query-gpu=memory.used` 判空闲（<100 MiB）。
- 运行中脚本不可 `tar xzf` 原地重写 → 按文件上传 + `mv` 替换。
- 上传排除规则只匹配顶层目录（`radar_nwp_bench/data/` 必须上传）；改文件须一起同步。
- SSH banner 偶尔丢失需重试 6–8 次；通道上限 ~3600 s → 长任务脱离通道。
- **stdout 重定向时 Python 块缓冲** → 脚本加 `reconfigure(line_buffering=True)`。
- paramiko 嵌套 heredoc 被 shell 引号吃掉 → 远端脚本用"写文件 + 单独执行"两步。
- **不要 `open('/dev/stdin').read()`**：永久阻塞。用独立的 `tools/_xxx.py`。
- `dataset[0]` 返回**单样本**（无 batch 维）；`x_nwp` 是 (2,27,H,W)，2 是时间片。
- 本地长任务勿用 `cmd | tail -N`（SIGTERM 时管道吞掉输出）→ 直接写文件。
- 量级问题先测量再判断（曾以为 GroupNorm 抹掉空间信息，实测 InstanceNorm 更差）。
- **远端 SSH 可能"banner 正常但密码被拒"** → 服务端拒绝了认证，非本地问题；
  此时若已把 `.eval.json` 拉回本地，分析照常可做。

## 性能（远程实测）

loader 8 ms/item（npz 762 ms）；GPU 60–74 %；1 epoch（46 200 样本，patch 64）327 s（独占）
/ 8 模型并行各占一卡 170–346 s；convlstm 430 s、predrnn 1060 s per epoch；
量子插件开销 +43 %/epoch；R1/R2 的 QPE 分支 126 s/epoch。

## 当前状态（2026-11-02 13:30）

- ✅ **score-aware 全矩阵**：18/18 × 20 epoch（本地副本 `tools/_evals/score_matrix/`）。
- ✅ **早停逐 epoch 曲线**：21 份（峰值 ep2，见第 25 条）。
- ✅ **持久性基线**：CSI@2.0 = 0.0924（第 24 条）。
- ✅ **基线零实验**：PASS（第 34 条）。
- ✅ **R1/R2 实现**：`y_hist` 数据层 + `simvp_v2_qpe_guided`（R1）+ `simvp_v2_assimilated`
  （R2）+ 分层损失适配器 + `--qpe-weight`；测试 546/546。
- 🔄 **运行中**：`runs/route_r1r2/` 7 臂（base_concat / qpe_0 / qpe_05 /
  assim_lag11 / lag4 / lag2 / lag0）+ `runs/route_baseline/` 2 模型，
  8 卡，20 epoch。监控自动化 `a91e11f5-07de-4920-abc5-9d9bf24b1619`。
- 权威结果：`docs/RESULTS_FINAL.md`（**第 10 节 = 全矩阵**）· `docs/THREE_DEFECTS.md`
  （根因）· `docs/MASTER.md`（主控）· `docs/SUGGESTION.md`（R1–R5 代码与实验规程）。
"""


def main() -> None:
    MEMORY.write_text(TEXT, encoding="utf-8")
    size = len(TEXT.encode("utf-8"))
    print(f"MEMORY.md rewritten: {size} bytes, {len(TEXT.splitlines())} lines")


if __name__ == "__main__":
    main()
