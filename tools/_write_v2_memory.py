"""Write the v2.0 master-document state into the project's long-term memory.

The memory already carries the findings. What is new in this version is
the set of routes, the two data facts the routes rest on, the six places
where the original sketch does not run against this repository, and the
hardened remote transport.
"""
import pathlib

MEMORY = pathlib.Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
ANCHOR = "## 远端操作坑（血泪）"

BLOCK = """## v2.0 主控文档与 R1-R5 路线（11-02 11:00）

**`docs/MASTER.md` 已升级到 v2.0（902 行）**：完整保留 §1-§13，新增
§14 深度调研（QPE-QPF 纠缠 → 训练后期退化）、§15 TGRS 三幕剧、§16 绝境迭代
方法论（残差分解 → Oracle 上界 → 正交文献 → 单变量验证）。
**`docs/SUGGESTION.md`** 承载 R1-R5 的代码骨架与分步实验规程。

### 新增两个关键数据事实（v2.0 的推理基石）
1. **RADAR 与 LABEL 时间戳 98/98 完全对齐** → 历史 11 帧的地面降水真值
   `y_hist` **逐帧可得**（R1/R2 的数据前提）。
2. **同帧雷达 CR 与地面 LABEL 逐像素相关仅 r = +0.17**（跨模态鸿沟）→
   "从雷达反演雨量"本身就是病态问题，与"未来平流外推"是两个子任务。

### 五条新路线
| 路线 | 内容 | 推理期用 y_hist |
|---|---|---|
| R1 | 历史 11 帧 LABEL 辅助 QPE 监督（LUPI，共享 RainHead）| **否**（零泄漏）|
| R2 | 历史 LABEL 显式同化 + 12/24 min 业务时滞模拟 | **是**（另一类设置，不可与纯雷达比）|
| R3 | 双轨 checkpoint（best.pt 由验证集选）+ 时效折现 | 否 |
| R4 | 可微半拉格朗日平流 + NWP 源汇残差（守恒 lambda<10km 谱）| 否 |
| R5 | 多尺度邻域软 FSS（1/3/5 窗口）+ 频域幅度谱约束 | 否 |

### ep16→ep19 退化的机制解释（**目前是推理，不是实测**）
端到端模型同时学两个子任务：QPE（零时滞反演）与 QPF（含位移混沌误差）。
远期位移的**双倍惩罚**（既算 FA 又算 Miss）迫使解码器在训练后期
**自我平滑以对冲惩罚** → 尖锐对流核心被抹平 → 与 §4.6 的 high_k 塌陷
（corr(high_k, CSI@2.0)=+0.950）独立吻合。**R1/R3 是它的直接检验对象。**

### Gemini 初稿的 6 处硬错误（照抄跑不起来，已在 SUGGESTION.md §0 标明）
1. `BalancedExtremePrecipLoss` 在 **`models/common.py`**，不在 `metrics/losses.py`
2. 该损失 `forward(pred, target, cum_probs=None)`：**无 kwargs、无
   _compute_core_loss、返回标量非 tuple**
3. score-aware 走 **`_BalancedExtremeCriterion` 适配器**，它**不转发 y_hist**
4. `simvp_v2_assimilated.forward` 加了 `y_hist` 参数 → **evaluate.py 调
   model(x,nwp) 拿不到** → R2 模型无法评测
5. `y_hist` 插值取 slot±1 → **连续两洞时退化为全零**（须向外找最近有效邻居）
6. 平流层的像素→归一化换算须用 **2/(W-1)**，不能写常数 0.05

### 远端 SSH 接口已固化（`tools/remote.py`）
- **失败分层诊断**：`--doctor` 逐层查 DNS → TCP → banner → 认证，**精确到层**
- **只重试可重试的**：认证/权限/磁盘**不重试**（重试可能锁账户）
- **人工介入卡片**：跨平台（PowerShell/cmd/bash）给出可执行的手动命令
- **本地改 → 同步 → 远端执行**：`--sync ... -- "<cmd>"` 一次连接完成
- **`--script X --detached LOG`** 绕开 ~3600 s 通道上限
- **初版有致命 bug**：引用了 paramiko 不存在的 `SFTPPermissionError` →
  classify() 对任何非认证失败抛 AttributeError。已修 + 10 项测试锁定。

"""


def main() -> None:
    text = MEMORY.read_text(encoding="utf-8")
    if "v2.0 主控文档与 R1-R5 路线" in text:
        print("already present")
        return
    if ANCHOR not in text:
        raise SystemExit(f"anchor not found: {ANCHOR!r}")
    MEMORY.write_text(text.replace(ANCHOR, BLOCK + ANCHOR, 1), encoding="utf-8")
    print("MEMORY.md updated")


if __name__ == "__main__":
    main()
