import os
import torch
import numpy as np
import csv
from torchvision import transforms
from PIL import Image

# 导入核心工具函数
from test_system1_protocol import (
    DinoV3Model, LinearProbe, REAL_CLASS_IDX, FAKE_CLASS_IDX,
    tensor_to_gray_uint8, validate_region_with_neighbor,
    cluster_patch_indices, cluster_to_bbox
)

# ===============================================================
# 全局阈值常量
# ===============================================================
SCORE_TH = 0.10
RATIO_TH = 1.00
TOKEN_TH = 0.05
GLOBAL_FAKE_GATE = 0.25  # Phase B.2: 软全局门控


def get_token_contrast(cluster, forensic_scores, num_prefix):
    """
    在 token-space（token空间）中，计算 cluster（候选簇）与其 ring（环邻域）的分数差。
    """
    rel_indices = [x[0] - num_prefix for x in cluster]
    num_patches = forensic_scores.shape[0]
    grid_size = int(np.sqrt(num_patches))

    cluster_set = set(rel_indices)
    ring_set = set()

    for idx in rel_indices:
        r = idx // grid_size
        c = idx % grid_size
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    n_idx = nr * grid_size + nc
                    if n_idx not in cluster_set:
                        ring_set.add(n_idx)

    if not ring_set:
        return 0.0

    cluster_mean = sum([forensic_scores[i].item() for i in cluster_set]) / len(cluster_set)
    ring_mean = sum([forensic_scores[i].item() for i in ring_set]) / len(ring_set)

    return cluster_mean - ring_mean


def build_evidence_objects(
    clusters,
    forensic_scores,
    gray_img,
    num_prefix,
    fake_prob,
    runtime_top_k_abs
):
    """
    构建：
    1. proposal_objects（所有候选对象，含 Rejected）
    2. validated_evidence_objects（正式验证通过的证据对象）
    3. evidence_coverage（证据覆盖度标量）
    """
    proposal_objects = []

    for region_id, cluster in enumerate(clusters):
        if len(cluster) < 2:
            continue

        bbox = cluster_to_bbox(cluster)
        _, contrast_ratio, _, _ = validate_region_with_neighbor(gray_img, bbox)

        patch_ids_abs = [x[0] for x in cluster]
        patch_ids_rel = [idx - num_prefix for idx in patch_ids_abs]

        patch_scores = [forensic_scores[idx - num_prefix].item() for idx in patch_ids_abs]
        score_max = max(patch_scores)
        score_mean = float(np.mean(patch_scores))

        token_contrast = get_token_contrast(cluster, forensic_scores, num_prefix)

        pixel_any_passed = score_max > SCORE_TH
        token_any_passed = (token_contrast > TOKEN_TH) if fake_prob > GLOBAL_FAKE_GATE else False
        is_validated = pixel_any_passed or token_any_passed
        rescued_by_token = (not pixel_any_passed) and token_any_passed
        high_conf = pixel_any_passed and (contrast_ratio > RATIO_TH)

        if high_conf:
            evidence_level = "HighConfidenceEvidence"
        elif is_validated:
            evidence_level = "WeakEvidence"
        else:
            evidence_level = "Rejected"

        proposal_objects.append({
            "region_id": region_id,
            "bbox_224": bbox,
            "patch_ids_abs": patch_ids_abs,
            "patch_ids_rel": patch_ids_rel,
            "num_patches": len(patch_ids_abs),
            "score_max": score_max,
            "score_mean": score_mean,
            "contrast_ratio": contrast_ratio,
            "token_contrast": token_contrast,
            "pixel_any_passed": pixel_any_passed,
            "token_any_passed": token_any_passed,
            "rescued_by_token": rescued_by_token,
            "is_validated": is_validated,
            "evidence_level": evidence_level
        })

    # 正式验证通过的证据对象
    validated_evidence_objects = [obj for obj in proposal_objects if obj["is_validated"]]

    # 排序：高置信 > 弱证据，再按 score / token / patch 数排序
    validated_evidence_objects = sorted(
        validated_evidence_objects,
        key=lambda obj: (
            int(obj["evidence_level"] == "HighConfidenceEvidence"),
            obj["score_max"],
            obj["token_contrast"],
            obj["num_patches"]
        ),
        reverse=True
    )

    # 写入 rank（排序名次）
    for rank, obj in enumerate(validated_evidence_objects, start=1):
        obj["rank"] = rank

    for obj in proposal_objects:
        if "rank" not in obj:
            obj["rank"] = -1

    # evidence_coverage = validated patch 数 / runtime top-k patch 数
    validated_patch_ids = set()
    for obj in validated_evidence_objects:
        validated_patch_ids.update(obj["patch_ids_abs"])

    top_k_patch_ids = set(runtime_top_k_abs)
    evidence_coverage = len(validated_patch_ids & top_k_patch_ids) / max(len(top_k_patch_ids), 1)

    return proposal_objects, validated_evidence_objects, evidence_coverage


def evaluate_folder_raw(
    folder_path,
    label,
    dataset_name,
    expected_count,
    model,
    linear_probe,
    official_fisher_indices,
    num_prefix,
    device,
    transform
):
    valid_exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(valid_exts)
    ])

    assert len(image_files) == expected_count, f"[!] 警告: {folder_path} 中的图片数量不匹配！"

    raw_results = []

    for img_path in image_files:
        raw_img = Image.open(img_path).convert('RGB')
        input_tensor = transform(raw_img).unsqueeze(0).to(device)
        gray_img = tensor_to_gray_uint8(input_tensor)

        with torch.no_grad():
            outputs = model(input_tensor, return_protocol=True)
            token_sequence = outputs["token_sequence"]

            # 分类概率
            selected_tokens = token_sequence[:, official_fisher_indices, :]
            probe_logits = linear_probe(selected_tokens)
            probs = torch.softmax(probe_logits, dim=-1)[0]
            fake_prob = probs[FAKE_CLASS_IDX].item()

            pred_label = "Fake" if fake_prob >= 0.5 else "Real"
            cls_correct = int(pred_label == label)

            # token-level suspiciousness scores（token级可疑度分数）
            patch_tokens = token_sequence[:, num_prefix:, :]
            w_real = linear_probe.fc.weight[REAL_CLASS_IDX]
            w_fake = linear_probe.fc.weight[FAKE_CLASS_IDX]
            delta_w_norm = torch.nn.functional.normalize(w_fake - w_real, dim=0)
            patch_norm = torch.nn.functional.normalize(patch_tokens[0], dim=-1)
            forensic_scores = torch.matmul(patch_norm, delta_w_norm)

            _, top_k_indices_patch = torch.topk(forensic_scores, k=min(10, forensic_scores.shape[0]))

        runtime_top_k_abs = (top_k_indices_patch + num_prefix).cpu().tolist()
        clusters = cluster_patch_indices(runtime_top_k_abs, num_prefix=num_prefix)

        num_raw_clusters = len(clusters)
        num_singletons = sum(1 for c in clusters if len(c) < 2)

        # 构建 proposal / validated evidence / coverage
        proposal_objects, validated_evidence_objects, evidence_coverage = build_evidence_objects(
            clusters=clusters,
            forensic_scores=forensic_scores,
            gray_img=gray_img,
            num_prefix=num_prefix,
            fake_prob=fake_prob,
            runtime_top_k_abs=runtime_top_k_abs
        )

        num_valid_clusters = len(validated_evidence_objects)

        pixel_any_passed = any(obj["pixel_any_passed"] for obj in proposal_objects)
        token_any_passed = any(obj["token_any_passed"] for obj in proposal_objects)
        has_any_passed = any(obj["is_validated"] for obj in proposal_objects)
        has_high_conf = any(obj["evidence_level"] == "HighConfidenceEvidence" for obj in validated_evidence_objects)
        rescued_by_token = any(obj["rescued_by_token"] for obj in validated_evidence_objects)

        if validated_evidence_objects:
            best_obj = validated_evidence_objects[0]
        elif proposal_objects:
            best_obj = sorted(
                proposal_objects,
                key=lambda obj: (obj["score_max"], obj["token_contrast"], obj["contrast_ratio"]),
                reverse=True
            )[0]
        else:
            best_obj = None

        best_score = best_obj["score_max"] if best_obj else 0.0
        best_ratio = best_obj["contrast_ratio"] if best_obj else 0.0
        best_token_contrast = best_obj["token_contrast"] if best_obj else 0.0

        if has_high_conf:
            status = "HighConfidenceEvidence"
        elif has_any_passed:
            status = "WeakEvidence"
        else:
            status = "NoEvidence"

        # source_type（来源类型）
        if dataset_name == "Group 1: GANs":
            source_type = "StyleGAN2" if label == "Fake" else "Real_Cars"
        elif dataset_name == "Group 2: Diffusion":
            source_type = "SDv1.5" if label == "Fake" else "ImageNet_Real"
        elif dataset_name == "Group 3a: Guided":
            source_type = "Guided" if label == "Fake" else "ImageNet_Real"
        elif dataset_name == "Group 3b: Midjourney":
            source_type = "Midjourney" if label == "Fake" else "ImageNet_Real"
        else:
            source_type = "Unknown"

        raw_results.append({
            "dataset": dataset_name,
            "source_type": source_type,
            "filename": os.path.basename(img_path),
            "label": label,
            "pred_label": pred_label,
            "cls_correct": cls_correct,
            "fake_prob": fake_prob,

            "num_raw_clusters": num_raw_clusters,
            "num_singletons": num_singletons,
            "num_valid_clusters": num_valid_clusters,

            "best_score_max": best_score,
            "best_contrast_ratio": best_ratio,
            "best_token_contrast": best_token_contrast,

            "pixel_any_passed": pixel_any_passed,
            "token_any_passed": token_any_passed,
            "has_any_passed": has_any_passed,
            "has_high_conf": has_high_conf,
            "rescued_by_token": rescued_by_token,

            "evidence_coverage": evidence_coverage,
            "proposal_objects": proposal_objects,
            "validated_evidence_objects": validated_evidence_objects,

            "status": status
        })

    return raw_results


def summarize_results(raw_results):
    stats = {
        "high_conf": 0,
        "any_passed": 0,
        "avg_prob": 0.0,
        "cls_correct_total": 0,
        "high_count": 0,
        "weak_count": 0,
        "none_count": 0,
        "avg_singletons": 0.0,
        "avg_token_contrast": 0.0,
        "rescued_count": 0,
        "avg_coverage": 0.0
    }

    if not raw_results:
        return stats

    for res in raw_results:
        stats["avg_prob"] += res["fake_prob"]
        stats["cls_correct_total"] += res["cls_correct"]
        stats["avg_singletons"] += res["num_singletons"]
        stats["avg_token_contrast"] += res["best_token_contrast"]
        stats["avg_coverage"] += res["evidence_coverage"]

        if res["has_any_passed"]:
            stats["any_passed"] += 1
        if res["has_high_conf"]:
            stats["high_conf"] += 1
        if res["rescued_by_token"]:
            stats["rescued_count"] += 1

        if res["status"] == "HighConfidenceEvidence":
            stats["high_count"] += 1
        elif res["status"] == "WeakEvidence":
            stats["weak_count"] += 1
        else:
            stats["none_count"] += 1

    n = len(raw_results)
    stats["avg_prob"] /= n
    stats["accuracy"] = stats["cls_correct_total"] / n
    stats["avg_singletons"] /= n
    stats["avg_token_contrast"] /= n
    stats["avg_coverage"] /= n
    return stats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 正在初始化 System 1 跨域软全局门控环境 ({device})...")

    model = DinoV3Model(model_name='dinov3_vit_7b', pool_type='patch_avg').to(device)
    model.eval()
    num_prefix = getattr(model.backbone, "num_prefix_tokens", 5)

    ckpt = torch.load('checkpoints/AIGCDetectionBenchmark/linear_probe.pth', map_location=device)
    official_fisher_indices = ckpt.get("token_indices", None)
    assert official_fisher_indices is not None and len(official_fisher_indices) > 0, "[!] 致命错误: token_indices 缺失！"

    probe_state = ckpt.get("probe_state_dict", ckpt.get("model_state_dict", None))
    assert probe_state is not None, "[!] 致命错误: probe_state_dict 缺失！"

    in_dim, out_dim = probe_state["fc.weight"].shape[1], probe_state["fc.weight"].shape[0]
    linear_probe = LinearProbe(input_dim=in_dim, num_classes=out_dim).to(device)
    linear_probe.load_state_dict(probe_state, strict=True)
    linear_probe.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    datasets = [
        ("Group 1: GANs", "batch_eval_total/batch_eval_gans/real/", "batch_eval_total/batch_eval_gans/fake/", 20),
        ("Group 2: Diffusion", "batch_eval_total/batch_eval_diffusion/real/", "batch_eval_total/batch_eval_diffusion/fake/", 20),
        ("Group 3a: Guided", "batch_eval_total/batch_eval_OOD/real/", "batch_eval_total/batch_eval_OOD/fake/", 10),
        ("Group 3b: Midjourney", "batch_eval_total/batch_eval_OOD1/real/", "batch_eval_total/batch_eval_OOD1/fake/", 10),
    ]

    all_raw_results = []

    print("\n================================================================================================================================================================")
    print("                                            System 1 跨域泛化能力测试 (Phase B.2: 软全局约束 0.25 + Evidence Objects/Coverage)                                  ")
    print("================================================================================================================================================================")
    print(f"补救门控: 仅当全局假图概率 Fake_Prob > {GLOBAL_FAKE_GATE:.2f} 且 语义级 Token_Contrast > {TOKEN_TH:.2f} 时，允许触发 Any Passed")
    print("-" * 160)
    print(f"{'数据集':<20} | {'Cls Acc (R/F)':<15} | {'Avg Prob (R/F)':<15} | {'Real (Any/High)':<15} | {'Fake (Any/High)':<18} | {'Fake (H/W/N)':<15} | {'AvgTok(F)':<10} | {'AvgCov(F)':<10} | {'Rescued(R/F)':<12}")
    print("-" * 160)

    for name, real_dir, fake_dir, expected_count in datasets:
        real_raw = evaluate_folder_raw(
            real_dir, "Real", name, expected_count,
            model, linear_probe, official_fisher_indices, num_prefix, device, transform
        )
        fake_raw = evaluate_folder_raw(
            fake_dir, "Fake", name, expected_count,
            model, linear_probe, official_fisher_indices, num_prefix, device, transform
        )

        all_raw_results.extend(real_raw)
        all_raw_results.extend(fake_raw)

        real_stats = summarize_results(real_raw)
        fake_stats = summarize_results(fake_raw)

        acc_str = f"{real_stats['accuracy']*100:.0f}% / {fake_stats['accuracy']*100:.0f}%"
        prob_str = f"{real_stats['avg_prob']:.2f} / {fake_stats['avg_prob']:.2f}"
        real_str = f"{real_stats['any_passed']:>2} / {real_stats['high_conf']:>2}"
        fake_str = f"{fake_stats['any_passed']:>2} / {fake_stats['high_conf']:>2}"
        status_str = f"{fake_stats['high_count']:>2}/{fake_stats['weak_count']:>2}/{fake_stats['none_count']:>2}"
        tok_str = f"{fake_stats['avg_token_contrast']:.3f}"
        cov_str = f"{fake_stats['avg_coverage']:.3f}"
        rescue_str = f"{real_stats['rescued_count']} / {fake_stats['rescued_count']}"

        print(f"{name:<20} | {acc_str:<15} | {prob_str:<15} | {real_str:<15} | {fake_str:<18} | {status_str:<15} | {tok_str:<10} | {cov_str:<10} | {rescue_str:<12}")

    print("================================================================================================================================================================\n")

    csv_path = "cross_domain_phaseB3.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Source_Type", "Filename", "Label", "Pred_Label", "Cls_Correct",
            "Fake_Prob", "Num_Raw_Clusters", "Num_Singletons", "Num_Valid_Clusters",
            "Best_Score_Max", "Best_Contrast_Ratio", "Best_Token_Contrast",
            "Pixel_Any_Passed", "Token_Any_Passed",
            "Has_Any_Passed", "Has_High_Conf", "Rescued_By_Token",
            "Evidence_Coverage", "Num_Validated_Evidence_Objects",
            "Status"
        ])
        for res in all_raw_results:
            writer.writerow([
                res["dataset"], res["source_type"], res["filename"], res["label"], res["pred_label"], res["cls_correct"],
                round(res["fake_prob"], 4), res["num_raw_clusters"], res["num_singletons"], res["num_valid_clusters"],
                round(res["best_score_max"], 4), round(res["best_contrast_ratio"], 4), round(res["best_token_contrast"], 4),
                res["pixel_any_passed"], res["token_any_passed"],
                res["has_any_passed"], res["has_high_conf"], res["rescued_by_token"],
                round(res["evidence_coverage"], 4), len(res["validated_evidence_objects"]),
                res["status"]
            ])

    print(f"[+] 已导出包含 validated evidence objects / evidence coverage 的结果至: {csv_path}")


if __name__ == "__main__":
    main()