"""Write the master-document protocol into the project's long-term memory.

The memory file already carries the findings. What it does not carry is the
rule that the findings are consulted before new work starts, which is what
stops a refuted idea from being run again under a different name. This
appends that rule, and points at the document that holds the rest.
"""
import pathlib

MEMORY = pathlib.Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
ANCHOR = "## 远端操作坑（血泪）"

TEXT = """## 主控文档与研究协议（11-02 起，所有任务的入口）

**`R:/1radarNWPprediction/radar-nwp-bench/docs/MASTER.md` 是本项目唯一权威入口。**
任何任务开始前先读它；任何产出先写回它，再写别处。

**四道强制关卡（协议）**：

| 关卡 | 动作 | 关键点 |
|---|---|---|
| Gate 0 | 读 MASTER.md | 读 §3 细节处理、§4 已做尝试、§9 待办、§10 结论分层 |
| Gate 1 | **排查** | 这个想法试过吗？结论是什么？**按机制比对，不按名词**；缺什么条件？缺就写待办，不硬上 |
| Gate 2 | 测试 | 判据**先写后跑**；对照只差一个变量；训练只在远端；出数前过**数据可信性七项** |
| Gate 3 | **纳入** | 写回 §4（动机→实现→结果→结论→状态）、更新 §9 待办、§10 结论分层、§12 变更日志加一行 |

**Gate 1 特别重要**：本项目已两次踩"换个说法是同一个东西"的坑
（逐像素非对称 BCE ≡ 分层损失，两次都失败）。
§10.2「已被推翻」列了 6 条**不可再写**的说法（强降水塌陷是方法族性质 /
物理外推碾压深度学习 / 所有方法都模糊化 / M1 0.152 超过物理外推 /
损失是瓶颈架构无关 / CSI@0.1 代价使损失站不住）。

**可执行的载体**：
- 技能 `research-master-doc`（`~/.workbuddy/skills/research-master-doc/`）——
  协议 + MASTER 模板 + `scripts/check_master.py`（结构检查，PASS/FAIL）。
- 已合并进 aihero 的 `research` 技能（"two modes"分支）。
- 项目测试 `tests/test_master_document.py`（8 项）—— 文档结构**被回归验证**，
  不是靠自觉。

"""

text = MEMORY.read_text(encoding="utf-8")
if "主控文档与研究协议" in text:
    print("already present")
else:
    assert ANCHOR in text, "anchor not found"
    text = text.replace(ANCHOR, TEXT + ANCHOR, 1)
    MEMORY.write_text(text, encoding="utf-8")
    print("MEMORY.md updated")
