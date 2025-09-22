# TSPN Hyperparameter Evaluation Guide

本指南说明如何使用新的超参数评估工具对 TSPN 进行对比学习、小样本和优化器配置的系统实验。所有网格默认基于 `TSPNContrastive` + `contrastive_classification` + `contrastive_trainer` 的组合，以便统一处理投影层与小样本采样；仅基线配置会回退到原始 `TSPN`/`Default_task`/`Default_trainer` 组合。

## 1. 环境准备

1. 激活项目虚拟环境，并确保依赖已安装：
   ```bash
   pip install -r requirements.txt
   ```
2. 如果需要指定 GPU，可在运行脚本前设置环境变量：
   ```bash
   export TSPN_EVAL_DEVICES=0,1  # 示例：使用第 0 和第 1 块 GPU
   ```
3. 所有实验输出默认写入 `save/hparam_eval/`，不需要提前创建。

## 2. 快速开始

运行统一入口脚本即可触发评估：
```bash
script/hparam_eval/run_tspn_hparam_eval.sh --sweep all --max-parallel 2
```
典型参数说明：
- `--sweep {contrastive|fewshot|optimizer|baseline|all}`：选择要执行的实验族。
- `--max-parallel N`：控制并行运行的进程数，单机默认顺序执行。
- `--timeout SEC`：设置单个实验的超时时间，0 表示不限制。
- `--rerun-failed`：自动重跑一次失败实验。
- `--dry-run`：只打印计划执行的命令，不真正启动实验。

## 3. 进阶选项

| 环境变量              | 作用                                           |
|-----------------------|------------------------------------------------|
| `PYTHON_BIN`          | 指定 Python 解释器路径                         |
| `CONFIG_ROOT`         | 自定义超参数网格配置目录                       |
| `OUTPUT_ROOT`         | 自定义评估结果根目录                          |
| `TSPN_EVAL_DEVICES`   | 转发给脚本的 `--devices` 参数                  |
| `TSPN_EVAL_PIPELINE`  | 指定 `main.py` 的 `--pipeline` 取值            |
| `TSPN_EVAL_TIMEOUT`   | 默认超时时间（秒），可被命令行重写             |

除环境变量外，脚本的其余参数都会完整透传给 `script/hparam_eval/tspn_hparam_eval.py`。

## 4. 输出结构

完成后将在 `save/hparam_eval/` 下生成结构化结果：
```
save/hparam_eval/
  ├── baseline/
  │   └── baseline_original/
  ├── contrastive/
  │   └── ...
  ├── fewshot/
  │   └── ...
  ├── optimizer/
  │   └── ...
  ├── summary_metrics.csv
  └── summary_report.md
```
- 每个运行目录包含 `train.log`、`resolved_config.yaml`、复制的 Lightning `lightning_logs/` 以及 `run_summary.json`。
- 顶层 `summary_metrics.csv` 汇总所有指标；`summary_report.md` 以 Markdown 表格形式展示结果和基线差异。

## 5. 调参与排错

- 如果日志目录缺失，确认 `save/` 目录有写权限，且 `environment.project` 名称不会与旧实验冲突。
- 如需快速验证，可使用 `--limit 2 --dry-run` 生成两条配置并查看生成的临时 YAML。
- 失败任务的标准输出会保存在各自目录的 `train.log` 中。

## 6. 高算力服务器运行建议

- 在集群或高算力服务器上，建议结合 `--max-parallel` 与作业调度器（如 Slurm）使用，保证 GPU 占用合理。
- 如需断点续跑，可在已有结果目录的基础上调用 `--rerun-failed`，脚本会重跑之前失败的配置并更新汇总报告。

以上即为 TSPN 超参数评估工具的使用说明，欢迎根据团队需求扩展 YAML 网格或脚本逻辑。
