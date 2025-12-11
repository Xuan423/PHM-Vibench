import argparse
import os
import pandas as pd
import torch
from pytorch_lightning import seed_everything

from src.configs.config_utils import merge_with_local_override, path_name, transfer_namespace
from src.utils.config_utils import parse_overrides, apply_overrides_to_config
from src.utils.utils import init_lab, close_lab, load_best_model_checkpoint
from src.data_factory import build_data
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer


def _run_stage2_kshot(model, dataloader, k_shot: int, metric: str = "cosine"):
    if k_shot <= 0 or not hasattr(model, "use_concept") or not model.use_concept:
        return None
    device = next(model.parameters()).device
    model.eval()
    support = {}
    queries = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            batch["x"] = x
            y = batch["y"].to(device)
            out = model.forward_with_batch(batch, epoch=999)
            c = out["c"]
            for i in range(c.size(0)):
                label = int(y[i].item())
                if support.get(label, []).__len__() < k_shot:
                    support.setdefault(label, []).append(c[i])
                else:
                    queries.append((c[i], label))
    if not support or not queries:
        return None
    proto_labels = sorted(support.keys())
    protos = []
    for lbl in proto_labels:
        protos.append(torch.stack(support[lbl], dim=0).mean(dim=0))
    protos = torch.stack(protos, dim=0)  # [P, d]
    correct = 0
    for q, lbl in queries:
        if metric == "euclidean":
            dist = torch.norm(protos - q.unsqueeze(0), dim=1)
            pred_idx = torch.argmin(dist)
        else:
            sim = torch.nn.functional.cosine_similarity(protos, q.unsqueeze(0), dim=1)
            pred_idx = torch.argmax(sim)
        pred_lbl = proto_labels[int(pred_idx.item())]
        if pred_lbl == lbl:
            correct += 1
    acc = correct / max(len(queries), 1)
    return {"k_shot": k_shot, "metric": metric, "accuracy": acc}


def pipeline(args):
    """TSPN-CL pipeline with concept head and optional Stage2 K-shot calibration."""
    config_path = args.config_path
    print(f"[INFO] 加载配置文件: {config_path}")
    configs = merge_with_local_override(config_path, getattr(args, "local_config", None))

    if hasattr(args, "override") and args.override:
        print(f"[INFO] 应用CLI override参数: {args.override}")
        overrides = parse_overrides(args.override)
        configs = apply_overrides_to_config(configs, overrides)
        print(f"[INFO] 已应用 {len(overrides)} 个override参数")

    required_sections = ["data", "model", "task", "trainer", "environment"]
    for section in required_sections:
        if not hasattr(configs, section):
            print(f"[ERROR] 配置文件中缺少 {section} 部分")
            return

    args_environment = transfer_namespace(configs.environment if hasattr(configs, "environment") else {})
    args_data = transfer_namespace(configs.data if hasattr(configs, "data") else {})
    args_model = transfer_namespace(configs.model if hasattr(configs, "model") else {})
    args_task = transfer_namespace(configs.task if hasattr(configs, "task") else {})
    args_trainer = transfer_namespace(configs.trainer if hasattr(configs, "trainer") else {})

    for key, value in configs.environment.__dict__.items():
        if key.isupper():
            os.environ[key] = str(value)
            print(f"[INFO] 设置环境变量: {key}={value}")

    all_results = []

    for it in range(args_environment.iterations):
        print(f"\n{'='*50}\n[INFO] 开始实验迭代 {it+1}/{args_environment.iterations}\n{'='*50}")
        path, name = path_name(configs, it)
        args_trainer.logger_name = name
        current_seed = args_environment.seed + it
        seed_everything(current_seed)
        print(f"[INFO] 设置随机种子: {current_seed}")
        init_lab(args_environment, args, name)

        print("[INFO] 构建数据工厂...")
        data_factory = build_data(args_data, args_task)
        print("[INFO] 构建模型...")
        model = build_model(args_model, metadata=data_factory.get_metadata())

        print("[INFO] 构建任务...")
        task = build_task(
            args_task=args_task,
            network=model,
            args_data=args_data,
            args_model=args_model,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=data_factory.get_metadata(),
        )

        print("[INFO] 构建训练器...")
        trainer = build_trainer(args_environment, args_trainer, args_data, path)

        print("[INFO] 开始训练...")
        trainer.fit(task, data_factory.get_dataloader("train"), data_factory.get_dataloader("val"))

        print("[INFO] 加载最佳模型并测试...")
        task = load_best_model_checkpoint(task, trainer)
        result = trainer.test(task, data_factory.get_dataloader("test"))
        data_factory.data.close()
        all_results.append(result[0])

        print("[INFO] 保存测试结果...")
        result_df = pd.DataFrame([result[0]])
        result_df.to_csv(os.path.join(path, f"test_result_{it}.csv"), index=False)

        # Stage2 K-shot calibration (optional)
        if getattr(args_model, "stage2_enable", False) and getattr(args_model, "stage2_k_shot", 0) > 0:
            print("[INFO] Stage2 K-shot 校准...")
            model.eval()
            kshot_res = _run_stage2_kshot(
                task.network,
                data_factory.get_dataloader("test"),
                k_shot=getattr(args_model, "stage2_k_shot", 0),
                metric=getattr(args_model, "prototype_metric", "cosine"),
            )
            if kshot_res:
                print(f"[INFO] Stage2 K-shot 结果: {kshot_res}")
                pd.DataFrame([kshot_res]).to_csv(os.path.join(path, f"kshot_result_{it}.csv"), index=False)
            else:
                print("[WARN] Stage2 K-shot 未生成结果（可能支持/查询不足）。")

        close_lab()

    print(f"\n{'='*50}\n[INFO] 所有实验已完成\n{'='*50}")
    pd.DataFrame(all_results).to_csv(os.path.join(path, "all_results.csv"), index=False)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSPN-CL pipeline")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/demo/01_cross_domain/X_DG/tspn_cl.yaml",
        help="配置文件路径",
    )
    parser.add_argument("--notes", type=str, default="", help="实验备注")
    parser.add_argument("--local_config", type=str, default=None, help="本机覆盖配置路径（可选）")
    args = parser.parse_args()
    results = pipeline(args)
    print("完成所有实验！")
