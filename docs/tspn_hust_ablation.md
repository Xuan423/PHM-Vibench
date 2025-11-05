# TSPN HUST Ablation Suite

本说明涵盖如何运行并扩展 `configs/experiments/tspn_hust_ablation/` 所提供的消融实验。若需同时运行 HUST 与 SDUST 的统一消融流程，请参考《docs/tspn_contrastive_ablation.md》。该套件基于 `configs/demo/X_Single_DG/TSPN_FewShot/contrastive.yaml`（迭代次数已统一设置为 5），用于评估对比损失分支、episode 采样策略以及无对比 baseline 的贡献。

## 目录结构

```
configs/experiments/tspn_hust_ablation/
  ├─ base.yaml                  # 共享入口，导入 demo 对比配置并写入公共 metadata
  ├─ contrastive_ssqq.yaml      # 仅保留 support-support / query-query 分支
  ├─ contrastive_qs.yaml        # 仅保留 support-query 分支
  ├─ contrastive_disabled.yaml  # 完全关闭对比损失
  ├─ no_episode.yaml            # 停用 episode，改为单分支标准训练
  ├─ support_no_align.yaml      # 保留 episode，关闭支持集原型对齐
  └─ baseline_tspn.yaml         # 无对比、无 episode 的基础 TSPN

script/hparam_eval/
  ├─ tspn_hust_ablation.py      # Python 调度脚本
  └─ run_tspn_hust_ablation.sh  # Shell 包装脚本（可选）
```

每个 YAML 在 `task.metadata` 中记录对应的 `variant_label`、`variant_type`、`episode_enabled`、`support_alignment`、`active_branches` 等信息，调度脚本会将其写入汇总表，方便论文/报告引用。

## 运行方式

### Python 模块

```bash
python -m script.hparam_eval.tspn_hust_ablation \
  --config-root configs/experiments/tspn_hust_ablation \
  --output-root save/hust_ablation \
  --max-parallel 2 \
  --device-pool 0,1 \
  --notes "hust ablation sweep"
```

可选参数：

- `--variants contrastive_full baseline_tspn`：只跑指定变体。
- `--dry-run`：生成配置与日志命令但不启动训练。
- `--devices 0,1`：指定 GPU。
- `--pipeline Pipeline_02_pretrain_fewshot`：自定义 pipeline。

### Shell 包装脚本

```bash
script/hparam_eval/run_tspn_hust_ablation.sh --max-parallel 2
```

或在运行前设置设备池，以便并行占用多个 GPU：

```bash
HUST_ABLATION_DEVICE_POOL="0,1" \
script/hparam_eval/run_tspn_hust_ablation.sh --max-parallel 2
```

支持的环境变量：

| 变量名                   | 说明                                           |
|--------------------------|------------------------------------------------|
| `CONFIG_ROOT`            | 指定配置目录（默认 `configs/experiments/tspn_hust_ablation`） |
| `OUTPUT_ROOT`            | 指定输出目录（默认 `save/hust_ablation`）     |
| `HUST_ABLATION_VARIANTS` | 以空格分隔的变体列表                           |
| `HUST_ABLATION_DEVICES`  | 转发给 `--devices`                             |
| `HUST_ABLATION_DEVICE_POOL` | 转发给 `--device-pool`，按 GPU 列表轮询      |
| `HUST_ABLATION_PIPELINE` | 转发给 `--pipeline`                            |
| `HUST_ABLATION_TIMEOUT`  | 转发给 `--timeout`                             |
| `HUST_ABLATION_NOTES`    | 附加备注，写入 `environment.notes`             |
| `HUST_ABLATION_STATS_METRIC_COLUMN` | 指定统计脚本的精度列名（可选）      |
| `HUST_ABLATION_STATS_METRIC_PATTERN` | 指定统计脚本的精度列匹配模式        |
| `HUST_ABLATION_STATS_OUTPUT_DIR` | 将统计结果写入的目录（默认同 OUTPUT_ROOT） |

## 产出

- 每个变体会在 `save/hust_ablation/<variant_name>/` 下生成：
  - `train.log`
  - `run_summary.json`（状态、元信息、关键指标）
  - `resolved_config.yaml`（最终训练配置）
- 顶层 `save/hust_ablation/` 会聚合：
  - `ablation_summary.csv` / `ablation_summary.md`

汇总表包含：

- `variant`、`status`、`runtime_sec`
- `meta_*` 字段（来自 `task.metadata`）
- 所有以 `test` 开头的指标（如 `test_acc_HUST`）

### 精度统计脚本

完成一轮实验后，可运行补充脚本快速整理精度指标：

```bash
python -m script.hparam_eval.tspn_hust_ablation_stats --input save/hust_ablation
```

默认读取 `save/hust_ablation/ablation_summary.csv`，自动识别包含 `test_acc` 的列作为精度字段（可通过 `--metric-column` 指定其他列名），并导出：

- `accuracy_records.csv`：逐次实验的精度记录及关键元信息。
- `accuracy_summary.csv`：按 `variant`、`contrastive_loss_weight`、`source_domain_tag` 等分组的精度统计（均值、标准差、最优 run 等）。

若将输出写到其他目录，可使用 `--output-dir <path>` 进行覆盖。

也可以直接通过 shell 脚本在同一次调用中完成实验与统计。例如：

```bash
HUST_ABLATION_DEVICE_POOL="0,1" \
HUST_ABLATION_STATS_OUTPUT_DIR="save/hust_ablation/stats" \
script/hparam_eval/run_tspn_hust_ablation.sh --max-parallel 2
```

上述命令会运行所有 ablation 变体，并在结束后生成精度报表。

## 扩展建议

1. **新增变体**：拷贝 `base.yaml`，在 `task.contrastive.active_branches` 或 `task.metadata` 中声明新组合，然后将其加入 `VARIANT_REGISTRY`。
2. **补充指标**：若 Lightning 导出的 `metrics.csv` 含更多自定义列，调度脚本会自动提取 `test*` 前缀的键；如需额外统计，可在 `summarise_results` 内扩展逻辑。
3. **可视化/分析**：使用 `ablation_summary.csv` 结合 `pandas`/`seaborn` 绘制对比分支、episode 方案之间的差异。

> 注意：该套件默认继承 demo 的数据路径与超参设置，仅在 `config_root` 中进行模块化开关。若需调整数据或模型结构，请在新 YAML 中添加对应的覆盖项。
