# radarNWP_NewTask

雷达 + NWP 降水临近预报统一基准的**代码库**（本仓库只收可执行代码与配置，不含实验记录文档）。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `radar_nwp_bench/` | 框架层：数据读取、指标、配置与模型接缝（`data/`、`metrics/`、`models/`） |
| `models/` | 40 个基线模型实现，每个模型一个包，由注册表发现而非直接导入 |
| `configs/` | `default.yaml`（共享默认值）+ `models/<name>.yaml`（逐模型超参）+ `data/` |
| `scripts/` | `train.py`、`evaluate.py`、`pack_data.py`、各阶段 Slurm 作业与提交脚本 |
| `tools/` | 诊断探针、远程通道 `remote.py`、判据读取器等一次性工具 |
| `tests/` | pytest 用例 |

## 环境

```bash
conda env create -f environment.yaml     # 或
pip install -e ".[dev]"
```

`environment.yaml` 对应 conda 环境名 `radar-nwp-bench`；集群上使用的环境名为 `radar_nwp`。
Python ≥ 3.10（已在 3.12 / torch 2.5.1 上验证）。

## 数据

数据**不入库**。`configs/default.yaml` 的 `data.root` 指向存放 `CP_*` case 目录的根路径；
若使用 `--packed`，则指向 `scripts/pack_data.py` 产出的 `.npz` 容器目录。
雷达 11 帧历史 + 10 帧预报，301×301 网格，6 分钟步长。

## 训练与评测

```bash
python scripts/train.py    --config configs/default.yaml --model <model_name> \
                           --data-root <DATA_ROOT> --out-dir runs/<exp> --epochs 10
python scripts/evaluate.py --config configs/default.yaml --model <model_name> \
                           --checkpoint <CKPT> --data-root <DATA_ROOT> --out-dir runs/<exp>
```

`--model` 取值见 `list_models()`（即 `models/` 下的包名）。
常用开关：`--packed`、`--patch-size`、`--slide-stride`、`--in-len/--out-len`、
`--seed`、`--device`（`auto` 时自动选 cuda）、`--log-every`。

## 远程工作流（Slurm 集群）

- **唯一通道**：`tools/remote.py`（SSH/SFTP）。同步与执行必须在同一次连接内完成：

  ```bash
  MSYS_NO_PATHCONV=1 python tools/remote.py --sync <file> -- '<remote command>'
  ```

  结果回拉用 `--download /tmp/x.json --out <本地路径>`（`--out` 必须显式写）。
  远端根目录由环境变量 `RADAR_ROOT` 控制（默认 `/home/wf/radarNWP`）。

- **长训练 / 评测**：写成 `.sbatch` 后用 `sbatch` 提交，不要用后台进程绕过队列；
  登录会话看不到 GPU。
  集群参数：partition `shared` / account `research` / qos `standard`，
  单作业 ≤6 GPU、≤96 CPU、≤384 GiB、≤30 h。
  日志目录 `run_logs/slurm/` 须在提交前创建。

- conda 激活与 `set -u` 冲突：顺序为 `set -eo pipefail` → `source conda.sh` →
  `conda activate <env>` → `set -u`。

## 测试

```bash
pytest tests/
```

## 关于本仓库的收录范围

按"只收代码"策略导出，以下内容**有意排除**：

- 实验记录文档（`docs/`）、每日记忆日志、`.md` 备份；
- 运行产物与数据（`tools/_evals/` 下的 JSON、模型权重、`runs/`、`data/`、日志）；
- 缓存（`__pycache__`、`.pytest_cache`）。

此外，`qlfn2` 系列脚本（`scripts/sbatch_qlfn2_f38.sbatch`、`scripts/run_qlfn2_f38.sh`、
`tools/judge_f38_nwp.py`）依赖另一条代码线里的 `qlfn2` 包，该包位于远端
`/home/wf/radarNWPQML`，**未包含在本仓库**，使用前需先获取。
