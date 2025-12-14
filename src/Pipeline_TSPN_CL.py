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


def _run_stage2_imprint(
    model,
    dataloader,
    k_shot: int,
    *,
    imprint_fusion: bool = True,
    proto_n0: float = 5.0,
    use_temperature: bool = True,
    temperature_init: float = 1.0,
):
    """Stage2 K-shot adaptation by weight imprinting/fusion on the existing classifier head.

    Constraint: reuse the same classifier module instance; update weights in-place.
    """
    if (
        k_shot <= 0
        or not hasattr(model, "encode_with_batch")
        or not hasattr(model, "get_active_linear_head")
    ):
        return None

    device = next(model.parameters()).device
    model.eval()

    support = {}
    queries = []
    head = None
    head_id = None

    with torch.no_grad():
        for batch in dataloader:
            batch["x"] = batch["x"].to(device)
            batch["y"] = batch["y"].to(device)
            out = model.encode_with_batch(batch, epoch=999, update_topk=False)
            c = out["c"]
            file_ids = out.get("file_ids", out.get("file_id", None))
            if head is None:
                head = model.get_active_linear_head(file_ids)
                head_id = id(head)
            for i in range(c.size(0)):
                label = int(batch["y"][i].item())
                if len(support.get(label, [])) < k_shot:
                    support.setdefault(label, []).append(c[i].detach())
                else:
                    queries.append((c[i].detach(), label))

    if head is None or not support or not queries:
        return None

    weight_before = head.weight.detach().clone()
    bias_before = head.bias.detach().clone() if head.bias is not None else None

    with torch.no_grad():
        for label, vecs in support.items():
            if label < 0 or label >= head.out_features:
                continue
            mu_tgt = torch.stack(vecs, dim=0).mean(dim=0)
            if imprint_fusion:
                alpha = float(len(vecs)) / float(len(vecs) + max(proto_n0, 1e-6))
                mu_tgt = (1.0 - alpha) * weight_before[label] + alpha * mu_tgt
            mu_tgt = torch.nn.functional.normalize(mu_tgt, dim=0)
            head.weight.data[label].copy_(mu_tgt)
            if head.bias is not None:
                head.bias.data[label].zero_()

        if use_temperature:
            model.logit_temperature = float(temperature_init)

    # Evaluate on cached queries using the same head instance
    T = float(getattr(model, "logit_temperature", 1.0)) if use_temperature else 1.0
    correct = 0
    for q, lbl in queries:
        logits = head(q.unsqueeze(0)) / max(T, 1e-6)
        pred = int(torch.argmax(logits, dim=1).item())
        correct += int(pred == lbl)
    acc = correct / max(len(queries), 1)

    head_same = id(head) == head_id
    changed = not torch.allclose(head.weight.detach(), weight_before)
    return {
        "k_shot": k_shot,
        "mode": "imprint",
        "imprint_fusion": bool(imprint_fusion),
        "proto_n0": float(proto_n0),
        "use_temperature": bool(use_temperature),
        "temperature": float(T),
        "accuracy": float(acc),
        "head_instance_unchanged": bool(head_same),
        "weights_changed": bool(changed),
        "num_support_classes": int(len(support)),
        "num_queries": int(len(queries)),
        "bias_updated": bool(bias_before is not None),
    }


def _run_stage2_tune_metric_and_imprint(
    model,
    dataloader,
    k_shot: int,
    *,
    steps: int = 100,
    base_lr: float = 1e-3,
    lr_scale: float = 0.1,
    lambda_prox: float = 1e-3,
    imprint_fusion: bool = True,
    proto_n0: float = 5.0,
    use_temperature: bool = True,
    temperature_init: float = 1.0,
):
    """Optional Stage2: tune metric head (V) on support with proximal constraint, then imprint."""
    if k_shot <= 0 or steps <= 0:
        return None
    if not hasattr(model, "metric") or not hasattr(model, "encode_with_batch") or not hasattr(model, "get_active_linear_head"):
        return None

    device = next(model.parameters()).device
    model.eval()

    # Pass 1: collect K-shot support samples per class.
    support_counts = {}
    support_x = []
    support_y = []
    support_file_ids = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"]
            y = batch["y"]
            file_id = batch.get("file_id")
            for i in range(x.size(0)):
                lbl = int(y[i].item())
                if support_counts.get(lbl, 0) >= k_shot:
                    continue
                support_counts[lbl] = support_counts.get(lbl, 0) + 1
                support_x.append(x[i].detach().cpu())
                support_y.append(lbl)
                if file_id is None:
                    support_file_ids.append(0)
                elif torch.is_tensor(file_id):
                    support_file_ids.append(int(file_id.view(-1)[i].item()))
                elif isinstance(file_id, (list, tuple)):
                    support_file_ids.append(int(file_id[i]))
                else:
                    support_file_ids.append(int(file_id))

    if not support_x:
        return None

    x_s = torch.stack(support_x, dim=0).to(device)
    y_s = torch.tensor(support_y, device=device, dtype=torch.long)
    file_ids_s = torch.tensor(support_file_ids, device=device, dtype=torch.long)
    support_batch = {"x": x_s, "y": y_s, "file_id": file_ids_s}

    # Resolve head instance and cache original head weights for fusion.
    with torch.no_grad():
        # Precompute masked backbone features once; Stage2 only tunes metric head.
        extras_s = model.encode_with_batch(support_batch, epoch=999, update_topk=False)
        head = model.get_active_linear_head(extras_s.get("file_ids"))
        head_id = id(head)
        w_src = head.weight.detach().clone()
        h_masked_s = extras_s.get("h_masked")
        use_cached_h = h_masked_s is not None
        if use_cached_h:
            h_masked_s = h_masked_s.detach()

        # Initial imprint before tuning (improves stability).
        proto = {}
        for lbl in sorted(set(support_y)):
            mask = y_s == lbl
            proto[int(lbl)] = extras_s["c"][mask].mean(dim=0)
        for lbl, mu_tgt in proto.items():
            alpha = 1.0
            if imprint_fusion:
                n_y = float(int((y_s == lbl).sum().item()))
                alpha = n_y / (n_y + max(proto_n0, 1e-6))
                mu_tgt = (1.0 - alpha) * w_src[lbl] + alpha * mu_tgt
            mu_tgt = torch.nn.functional.normalize(mu_tgt, dim=0)
            head.weight.data[lbl].copy_(mu_tgt)
            if head.bias is not None:
                head.bias.data[lbl].zero_()
        if use_temperature:
            model.logit_temperature = float(temperature_init)

    # Freeze everything except metric head parameters.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.metric.parameters():
        p.requires_grad = True
    for p in head.parameters():
        p.requires_grad = False

    V0 = model.metric.weight.detach().clone()
    lr = float(base_lr) * float(lr_scale)
    T = max(float(getattr(model, "logit_temperature", 1.0)), 1e-6)

    # Stage2 tuning loop (support only) - manual SGD update to stay CPU-safe.
    for _ in range(int(steps)):
        for p in model.metric.parameters():
            p.grad = None
        if use_cached_h:
            c_s = model.metric(h_masked_s)
        else:
            extras_s = model.encode_with_batch(support_batch, epoch=999, update_topk=False)
            c_s = extras_s["c"]
        logits = head(c_s) / T
        loss_ce = torch.nn.functional.cross_entropy(logits, y_s)
        loss_prox = torch.mean((model.metric.weight - V0) ** 2)

        loss = loss_ce + float(lambda_prox) * loss_prox

        # Inherit Stage1 sparsity config only (no new Stage2 schedule).
        if bool(getattr(model, "sparsity_enable", False)):
            V = model.metric.weight
            row_c = float(getattr(model, "sparsity_coeff_row", 0.0))
            col_c = float(getattr(model, "sparsity_coeff_col", 0.0))
            if row_c > 0:
                loss = loss + row_c * torch.norm(V, p=1, dim=1).mean()
            if col_c > 0:
                loss = loss + col_c * torch.norm(V, p=2, dim=0).mean()

        loss.backward()
        with torch.no_grad():
            for p in model.metric.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)

    # Re-imprint with updated metric.
    with torch.no_grad():
        if use_cached_h:
            c_s = model.metric(h_masked_s)
        else:
            extras_s = model.encode_with_batch(support_batch, epoch=999, update_topk=False)
            c_s = extras_s["c"]
        proto = {}
        for lbl in sorted(set(support_y)):
            mask = y_s == lbl
            proto[int(lbl)] = c_s[mask].mean(dim=0)
        for lbl, mu_tgt in proto.items():
            alpha = 1.0
            if imprint_fusion:
                n_y = float(int((y_s == lbl).sum().item()))
                alpha = n_y / (n_y + max(proto_n0, 1e-6))
                mu_tgt = (1.0 - alpha) * w_src[lbl] + alpha * mu_tgt
            mu_tgt = torch.nn.functional.normalize(mu_tgt, dim=0)
            head.weight.data[lbl].copy_(mu_tgt)
            if head.bias is not None:
                head.bias.data[lbl].zero_()

    # Pass 2: evaluate on queries (skip first K per class as support).
    support_counts = {}
    correct = 0
    total = 0
    T = float(getattr(model, "logit_temperature", 1.0)) if use_temperature else 1.0
    with torch.no_grad():
        for batch in dataloader:
            batch["x"] = batch["x"].to(device)
            batch["y"] = batch["y"].to(device)
            extras = model.encode_with_batch(batch, epoch=999, update_topk=False)
            logits = head(extras["c"]) / max(T, 1e-6)
            preds = torch.argmax(logits, dim=1)
            for i in range(preds.size(0)):
                lbl = int(batch["y"][i].item())
                if support_counts.get(lbl, 0) < k_shot:
                    support_counts[lbl] = support_counts.get(lbl, 0) + 1
                    continue
                total += 1
                correct += int(int(preds[i].item()) == lbl)

    acc = correct / max(total, 1)
    return {
        "k_shot": int(k_shot),
        "mode": "tune_metric+imprint",
        "steps": int(steps),
        "lr_scale": float(lr_scale),
        "lambda_prox": float(lambda_prox),
        "imprint_fusion": bool(imprint_fusion),
        "proto_n0": float(proto_n0),
        "use_temperature": bool(use_temperature),
        "temperature": float(T),
        "accuracy": float(acc),
        "head_instance_unchanged": bool(id(head) == head_id),
    }


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
            stage2_mode = getattr(args_model, "stage2_mode", "imprint")
            if stage2_mode != "imprint":
                print(f"[WARN] 未知 stage2_mode={stage2_mode}，跳过 Stage2。")
                kshot_res = None
            else:
                if getattr(args_model, "stage2_tune_metric", False):
                    kshot_res = _run_stage2_tune_metric_and_imprint(
                        task.network,
                        data_factory.get_dataloader("test"),
                        k_shot=getattr(args_model, "stage2_k_shot", 0),
                        steps=getattr(args_model, "stage2_steps", 100),
                        base_lr=getattr(args_task, "lr", 1e-3),
                        lr_scale=getattr(args_model, "stage2_lr_scale", 0.1),
                        lambda_prox=getattr(args_model, "stage2_lambda_prox", 1e-3),
                        imprint_fusion=getattr(args_model, "stage2_imprint_fusion", True),
                        proto_n0=getattr(args_model, "stage2_proto_n0", 5.0),
                        use_temperature=getattr(args_model, "stage2_use_temperature", True),
                        temperature_init=getattr(args_model, "stage2_temperature_init", 1.0),
                    )
                else:
                    kshot_res = _run_stage2_imprint(
                        task.network,
                        data_factory.get_dataloader("test"),
                        k_shot=getattr(args_model, "stage2_k_shot", 0),
                        imprint_fusion=getattr(args_model, "stage2_imprint_fusion", True),
                        proto_n0=getattr(args_model, "stage2_proto_n0", 5.0),
                        use_temperature=getattr(args_model, "stage2_use_temperature", True),
                        temperature_init=getattr(args_model, "stage2_temperature_init", 1.0),
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
