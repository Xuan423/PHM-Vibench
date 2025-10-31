# TSPN Contrastive Sweep Guide

本指南说明如何运行精简后的 TSPN 对比式小样本超参数搜索。新的流程仅维护一份网格 `contrastive_grid.yaml`，围绕下列核心参数展开：

- `contrastive_loss_weight`
- `domains_per_episode`
- `classes_per_domain`
- `support_per_class`
- `query_per_class`

所有实验基于 `configs/demo/X_Single_DG/TSPN_FewShot/contrastive.yaml` 导入的默认配置，确保搜索结果可直接与官方 Demo 对齐。

## 1. 环境准备

1. 激活项目虚拟环境并安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 可选：通过环境变量指定 GPU：
   ```bash
   export TSPN_EVAL_DEVICES=0,1
   ```
3. 输出默认写入 `save/hparam_eval/`，无需手动创建目录。

## 2. 快速开始

执行入口脚本即可生成所有组合：
```bash
script/hparam_eval/run_tspn_hparam_eval.sh --max-parallel 2
```
常用参数：
- `--max-parallel N`：并行运行的进程数，默认串行执行。
- `--timeout SEC`：单个实验的超时时间，0 表示无限制。
- `--rerun-failed`：在首次 sweep 结束后重试失败任务。
- `--limit K`：仅生成前 `K` 条组合，便于调试。
- `--dry-run`：只打印计划命令，不实际启动训练。

脚本接受的其他参数会透传给 Python 模块 `script.hparam_eval.tspn_hparam_eval`。

## 3. 环境变量

| 变量名                | 说明                                                         |
|-----------------------|--------------------------------------------------------------|
| `PYTHON_BIN`          | 指定 Python 解释器（默认使用当前环境中的 `python`/`python3`） |
| `CONFIG_ROOT`         | 自定义网格配置目录，默认为 `configs/experiments/tspn_hparam_eval` |
| `OUTPUT_ROOT`         | 自定义输出根目录，默认为 `save/hparam_eval`                 |
| `TSPN_EVAL_DEVICES`   | 作为 `--devices` 传递给 Python 脚本                          |
| `TSPN_EVAL_PIPELINE`  | 传递给 `main.py` 的 `--pipeline`                             |
| `TSPN_EVAL_TIMEOUT`   | 默认超时时间（秒），可被命令行参数覆盖                       |
| `TSPN_EVAL_NOTES`     | 附加备注，将写入 `environment.notes`                         |

## 4. 输出结构

运行完成后可在 `save/hparam_eval/` 下看到如下目录：
```
save/hparam_eval/
  ├── contrastive/
  │   └── contrastive__contrastive_loss_weight_0p1__domains_per_episode_3__...
  │       ├── train.log
  │       ├── resolved_config.yaml
  │       ├── lightning_logs/…
  │       └── run_summary.json
  ├── contrastive_sweep_summary.csv
  └── contrastive_sweep_summary.md
```
- 每个运行目录包含训练日志、解析后的配置文件以及 Lightning 日志拷贝。
- 顶层 `contrastive_sweep_summary.csv`/`.md` 汇总每条组合的超参数、`test_acc`、运行状态与时长。

## 5. 调参与排错建议

- 使用 `--dry-run` 搭配 `--limit`，可快速检查生成的覆盖配置。
- 若日志未同步到 `save/`，确认运行账户对该目录拥有写权限。
- 失败运行的详细错误位于对应目录的 `train.log`。
- 汇总表中的 `test_acc` 若为 `NaN`，表示该运行未产出有效测试指标（可能因失败或超时）。

## 6. 高算力服务器提示

- 建议结合调度器设定 `--max-parallel`，避免 GPU 资源冲突。
- 通过 `--rerun-failed` 可以在修复故障后重跑失败配置并自动更新汇总表。

如需扩展搜索范围，可在现有 YAML 基础上增补新的取值列表；保持字段命名与脚本解析逻辑一致即可。
