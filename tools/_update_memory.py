import pathlib

p = pathlib.Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
lines = p.read_text(encoding="utf-8").splitlines()

anchor = None
for i, line in enumerate(lines):
    if "先怀疑构造" in line:
        anchor = i
        break
assert anchor is not None, "anchor not found"

end = anchor
for i in range(anchor, len(lines)):
    stripped = lines[i].lstrip()
    if stripped and stripped[0].isdigit() and "." in stripped[:4]:
        end = i
    elif stripped and not stripped.startswith((" ", "\t")) and end > anchor:
        break

addition = """
32. **✅✅ 决定性结论（11-01 第四轮）："强降水塌陷"是损失的性质，不是方法族的性质。**
    `docs/RESULTS_FINAL.md` 是权威记录（取代 TGRS_TABLES.md 的强降水结论）。
    **干预实验**：同一个骨干 simvp_v2（无分层头、无 NWP、无插件），唯一改变是损失：
    | | 损失 | CSI@2.0 | CSI@4.0 | high_k | BIAS@0.5 |
    |---|---|---|---|---|---|
    | 骨干 | 标准 | 0.004 | 0.000 | 0.0163 | 0.111 |
    | 骨干 | **score-aware** | **0.118** | **0.063** | **3.7388** | 0.630 |
    **CSI@2.0 30 倍、high_k 229 倍（超过物理外推 3.566）、CSI@4.0 从 0 到 0.063。**
    → ❌ "深度方法欠预测强降水是结构性缺陷" **错**
    → ❌ "物理外推在强降水上碾压深度学习" **错**（那是标准损失的效果）
    → ✅ "log 空间回归目标函数使所有方法欠预测 + 抹平小尺度 + 强降水塌陷"
33. **⚠️ 量子插件的最终判决（有干净对照）**：score-aware 损失下
    插件 CSI@2.0 = **0.008** vs 骨干 0.118（比值 0.07），比标准损失下的
    相对表现更差。**插件在正确损失下也没有增益。**
    最强配置 = **M1 concat + score-aware**：CSI@2.0 **0.152** > 物理外推 0.101，
    CSI@4.0 **0.088** > 0.035，CSI@0.1 保持 0.536。
34. **⚠️ 运行间方差不可忽略**：骨干+标准损失，旧矩阵 0.004 vs 修复管线后重跑 0.000
    （hits 29715 → 277）；CSI@0.5 0.060 vs 0.095。差异来自**数据管线修复**
    （标准损失下更保守）。→ 论文任何单次运行必须配 bootstrap CI + 多 seed。
35. **⚠️ 强降水得分是"用虚警换召回"**：C1 的 lift over 到处乱报 = **10.9×**
    （排除退化解），但**每窗口 precision 中位数 = 0.0000**（pysteps 是 0.0316）。
    → 论文只能写"更高召回 + 更高虚警率"，**不能写"更准确定位了强降水"**。
36. **⚠️ 远端坑（新增）**：用 paramiko + `setsid bash -c` 启动脚本时，
    **命令可能被执行两次**（两个实例写同一输出目录 → 4 个模型跑两遍）。
    症状：nvidia-smi 显示 8 个 compute app 但只有 4 个卡被派发。
    排查：`ps -eo pid,pgid,etime,args | grep '[r]un_xxx.sh'` 看 pgid，
    杀掉旧的进程组（`kill -- -<pgid>`）。"""

out = "\n".join(lines[: end + 1]) + "\n" + addition + "\n" + "\n".join(lines[end + 1:])
p.write_text(out, encoding="utf-8")
print("MEMORY.md updated")
