"""Write this round's findings into the project's long-term memory.

Two measured facts change how the results have to be read: the
early-stopping peak is at epoch two rather than sixteen, and the
persistence reference puts the floor for the heavy rain at 0.0924
rather than zero. Both are now in the master document; this puts the
same two into the memory, because the memory is what the next session
reads first.
"""
import pathlib

MEMORY = pathlib.Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
ANCHOR = "## 远端操作坑（血泪）"

BLOCK = """## ⚠️ 两条改变论断的实测（11-02 12:30，务必先读）

### ① 早停扫描已跑完：峰值在 **epoch 2**，不是 epoch 16

`runs/epoch_curve/`（simvp_v2 + score-aware，seed 0，n=538，逐 epoch 评测，21 份）：

| epoch | CSI@0.1 | CSI@0.5 | **CSI@2.0** | CSI@4.0 | BIAS@0.5 | hits@2.0 |
|---|---|---|---|---|---|---|
| 0 | 0.551 | 0.242 | **0.129** | 0.077 | 0.642 | 1 347 440 |
| 1 | 0.536 | 0.287 | **0.138** | 0.076 | **0.973** | 1 560 504 |
| **2** | 0.442 | 0.279 | **0.153** | **0.088** | 0.712 | 1 467 832 |
| 3 | 0.466 | 0.121 | 0.050 | 0.030 | 0.274 | 401 949 |
| 13 | 0.365 | 0.068 | 0.016 | 0.002 | 0.128 | 120 769 |
| 16 | 0.434 | 0.148 | 0.035 | 0.005 | 0.304 | 266 152 |
| 19 | 0.415 | 0.109 | 0.027 | 0.005 | 0.223 | 208 743 |
| 19（标准损失对照） | 0.512 | 0.086 | 0.004 | 0.000 | 0.154 | 29 010 |

- **峰值 ep2 = 0.1534，是末轮的 38.9 倍**；CSI@4.0 峰值 0.0884 是末轮的 629 倍。
- 旧文档说的 "ep16 = 0.1183" 来自 **D1 那个半训练 run**，不是这条曲线。
- **ep1 的 BIAS = 0.973（≈1.0）= 持久性的指纹** → 早期模型实质在做 QPE+持久性。
  → §14.1 的"双倍惩罚"解释方向对，但 ep2 的 0.153 只比持久性高 66 %，
  **必须用"真值雨量做持久性"的 Oracle 上界分离**（§16 第二步）。
- **运行方差**：同配置两次 ep19 = 0.0384 vs 0.027（**1.4 倍**）；
  而标准损失两次 0.0040 vs 0.0039（几乎一致）→ **score-aware 的方差大得多**，
  判据必须留出 1.4 倍余量，Δ 小于此不能称为提升。

### ② 持久性外推基线：强降水的地板是 **0.0924**，不是 0

`tools/_persistence_reference.py`（免训练，同 dataset/verifier/缝合，n=538，
Z-R 与 pysteps_adv 相同）：

| 参考 | CSI@0.1 | CSI@0.5 | **CSI@2.0** | CSI@4.0 |
|---|---|---|---|---|
| **持久性外推（复制最后一帧）** | 0.4470 | **0.2102** | **0.0924** | 0.0291 |
| pysteps_adv | 0.452 | 0.218 | 0.101 | 0.035 |
| score-aware 矩阵中位（17 模型） | 0.333 | 0.116 | 0.0262 | 0.0078 |
| score-aware 最好（mmf_rnn） | 0.295 | 0.171 | 0.0896 | 0.0373 |

- **18 个深度模型在强降水上没有一个超过"把最后一帧原样复制"**（最好的 0.0896 < 0.0924）。
- **pysteps_adv 相对持久性只高 9 %** —— 原先"物理外推碾压深度学习"的说法，
  实际上大半是在说持久性。**主表必须加持久性这一行。**
- **任何新方法的 CSI@2.0 必须 > 0.0924 才算有信号。**

### ③ 基线零实验：数据层加 y_hist **没有改变既有基线**（VERDICT: PASS）

用新代码重评既有 checkpoint（权重固定 → 无训练方差 → 精确检验）：
simvp_v2 的 CSI@0.1/0.5/2.0/4.0 = 0.4567/0.1437/0.0384/0.0078，与矩阵**逐位一致**，
n=538 一致，**命中数完全相同**；simvp_v2_nwp_concat 同样一致（0.4494/0.1225/0.0258/0.0047）。
→ 旧矩阵的基线仍然可用，**不必全基线重跑**；新实验用同一次运行的 `base_concat` 臂更干净。

"""


def main() -> None:
    text = MEMORY.read_text(encoding="utf-8")
    if "两条改变论断的实测" in text:
        print("already present")
        return
    if ANCHOR not in text:
        raise SystemExit(f"anchor not found: {ANCHOR!r}")
    MEMORY.write_text(text.replace(ANCHOR, BLOCK + ANCHOR, 1), encoding="utf-8")
    print("MEMORY.md updated")


if __name__ == "__main__":
    main()
