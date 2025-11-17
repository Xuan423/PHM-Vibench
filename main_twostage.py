import argparse
import importlib

from src.configs.config_utils import load_config


def main():
    """
    专用于两阶段流水线的入口，支持自动读取配置中的 pipeline.module 与 options。
    """
    parser = argparse.ArgumentParser(description="两阶段任务流水线")

    parser.add_argument(
        '--config_path',
        type=str,
        default='configs/demo/X_Single_DG/TSPN_FewShot/tspn_two_stage.yaml',
        help='配置文件路径',
    )
    parser.add_argument('--notes', type=str, default='', help='实验备注')
    parser.add_argument('--fs_config_path', type=str, default=None, help='few-shot config for pretrain pipeline')
    parser.add_argument(
        '--pipeline',
        type=str,
        default=None,
        help='实验流水线模块路径（默认按配置文件中的 pipeline.module 自动推断）',
    )
    parser.add_argument(
        '--two_stage_mode',
        type=str,
        default='auto',
        choices=['stage1', 'stage2', 'both', 'auto'],
        help='针对多阶段流水线：Stage1/Stage2/both 或 auto',
    )
    parser.add_argument('--stage1_checkpoint', type=str, default=None, help='Stage2 直接运行时使用的 checkpoint')
    parser.add_argument('--stage2_template', type=str, default=None, help='Stage2 自定义 YAML 模板')
    parser.add_argument('--projector_log_dir', type=str, default=None, help='覆写 projector 轨迹输出目录')
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录路径，会覆盖配置文件中的 data_dir 设置')
    parser.add_argument(
        '--set',
        action='append',
        metavar='KEY=VALUE',
        help='覆盖配置参数，支持点符号(例: --set model.lr=0.001 --set task.epochs=100)',
    )

    args = parser.parse_args()

    pipeline_name = _resolve_pipeline_module(args)
    args.pipeline = pipeline_name
    import_path = pipeline_name if pipeline_name.startswith("src.") else f"src.{pipeline_name}"
    pipeline = importlib.import_module(import_path)
    results = pipeline.pipeline(args)
    print("完成所有实验！")
    return results


def _resolve_pipeline_module(args):
    """根据配置和 CLI 参数推断流水线模块。"""

    module = getattr(args, "pipeline", None)
    if module and module.lower() != "auto":
        return module

    config_path = getattr(args, "config_path", None)
    if not config_path:
        return "Pipeline_01_default"

    try:
        config_preview = load_config(config_path)
    except Exception:
        return "Pipeline_01_default"

    pipeline_cfg = getattr(config_preview, "pipeline", None)
    if pipeline_cfg is None:
        return "Pipeline_01_default"

    options = getattr(pipeline_cfg, "options", None)
    if options is not None:
        for key, value in options.__dict__.items():
            if hasattr(args, key) and getattr(args, key, None) in (None, "", "auto"):
                setattr(args, key, value)

        enable_stage1 = getattr(options, "enable_stage1", None)
        if enable_stage1 is not None and getattr(args, "two_stage_mode", None) in (None, "", "auto"):
            args.two_stage_mode = "both" if enable_stage1 else "stage2"

        default_stage1_ckpt = getattr(options, "stage1_checkpoint", None)
        if getattr(args, "stage1_checkpoint", None) in (None, "", "auto") and default_stage1_ckpt:
            args.stage1_checkpoint = default_stage1_ckpt

    module_name = getattr(pipeline_cfg, "module", None)
    return module_name or "Pipeline_01_default"


if __name__ == "__main__":
    main()
