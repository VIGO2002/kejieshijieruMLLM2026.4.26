"""
build_system1_scan_table.py

功能：
1. 自动遍历多个候选数据集目录
2. 兼容两类目录结构：
   - dataset/0_real, dataset/1_fake
   - dataset/class_name/0_real, dataset/class_name/1_fake
3. 对候选图片只跑 System 1（不做完整导出）
4. 自动生成预扫描总表 CSV
5. 方便后续按 status / label 自动采样，构建 train_pool / holdout

说明：
- 这是“预扫描脚本”，不是最终导出脚本
- 目标是先自动知道哪些图片属于 No/Weak/High
"""

import os
import csv
import glob
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from PIL import Image
from torchvision import transforms

from test_system1_protocol import (
    DinoV3Model,
    LinearProbe,
    REAL_CLASS_IDX,
    FAKE_CLASS_IDX,
    tensor_to_gray_uint8,
    validate_region_with_neighbor,
    cluster_patch_indices,
    cluster_to_bbox,
)

# ============================================================
# 默认配置
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_OUTPUT_CSV = "system1_scan_table.csv"
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_PER_LEAF_PER_LABEL = 50   # 每个叶子目录中，每个标签最多抽多少张
DEFAULT_SCORE_TH = 0.10
DEFAULT_RATIO_TH = 1.00
DEFAULT_TOKEN_TH = 0.05

# 你当前环境里已有的 checkpoint
DEFAULT_LINEAR_PROBE_CKPT = "checkpoints/AIGCDetectionBenchmark/linear_probe.pth"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])

VALID_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp", ".WEBP")


# ============================================================
# 数据根目录配置
# 你可以按需继续加
# ============================================================

DATASET_ROOTS = [
    "/root/autodl-tmp/datasets/CNNDetection/biggan",
    "/root/autodl-tmp/datasets/CNNDetection/progan",
    "/root/autodl-tmp/datasets/CNNDetection/cyclegan",
    "/root/autodl-tmp/datasets/CNNDetection/gaugan",
    "/root/autodl-tmp/datasets/CNNDetection/stylegan",
    "/root/autodl-tmp/datasets/CNNDetection/stylegan2",
    "/root/autodl-tmp/datasets/Diffusion/glide_100_10",
    "/root/autodl-tmp/datasets/Diffusion/glide_100_27",
    "/root/autodl-tmp/datasets/Diffusion/glide_50_27",
    "/root/autodl-tmp/datasets/Diffusion/guided",
    "/root/autodl-tmp/datasets/Diffusion/ldm_100",
    "/root/autodl-tmp/datasets/Diffusion/ldm_200",
    "/root/autodl-tmp/datasets/Diffusion/ldm_200_cfg",
    "/root/autodl-tmp/datasets/Diffusion/pndm",
    "/root/autodl-tmp/datasets/Diffusion/vqdiffusion",
]


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def is_image_file(path: str) -> bool:
    return path.endswith(VALID_EXTS)


def list_images_in_dir(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    files = []
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        if os.path.isfile(p) and is_image_file(p):
            files.append(p)
    return sorted(files)


def sample_paths(paths: List[str], max_n: int, seed: int) -> List[str]:
    if len(paths) <= max_n:
        return paths
    rng = random.Random(seed)
    return sorted(rng.sample(paths, max_n))


def ensure_parent_dir(path: str):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


# ============================================================
# 自动发现 0_real / 1_fake 的叶子目录
# ============================================================

def find_label_dirs(root: str) -> List[Dict[str, str]]:
    """
    返回每个“叶子组”：
    {
        "dataset_root": root,
        "subset_name": "biggan" 或 "progan/airplane" 这类描述,
        "real_dir": ".../0_real",
        "fake_dir": ".../1_fake"
    }
    """
    results = []

    # 情况 A：root/0_real, root/1_fake
    real_dir = os.path.join(root, "0_real")
    fake_dir = os.path.join(root, "1_fake")
    if os.path.isdir(real_dir) and os.path.isdir(fake_dir):
        results.append({
            "dataset_root": root,
            "subset_name": os.path.basename(root),
            "real_dir": real_dir,
            "fake_dir": fake_dir,
        })
        return results

    # 情况 B：root/*/0_real, root/*/1_fake
    for child in sorted(os.listdir(root)):
        child_path = os.path.join(root, child)
        if not os.path.isdir(child_path):
            continue
        real_dir = os.path.join(child_path, "0_real")
        fake_dir = os.path.join(child_path, "1_fake")
        if os.path.isdir(real_dir) and os.path.isdir(fake_dir):
            results.append({
                "dataset_root": root,
                "subset_name": f"{os.path.basename(root)}/{child}",
                "real_dir": real_dir,
                "fake_dir": fake_dir,
            })

    return results


# ============================================================
# System 1 初始化
# ============================================================

def load_system1():
    print(f"[*] 正在初始化 System 1 ({DEVICE}) ...")
    model = DinoV3Model(model_name="dinov3_vit_7b", pool_type="patch_avg").to(DEVICE)
    model.eval()

    num_prefix = getattr(model.backbone, "num_prefix_tokens", 5)

    ckpt = torch.load(DEFAULT_LINEAR_PROBE_CKPT, map_location=DEVICE)
    official_fisher_indices = ckpt.get("token_indices", None)
    probe_state = ckpt.get("probe_state_dict", ckpt.get("model_state_dict", None))

    linear_probe = LinearProbe(
        input_dim=probe_state["fc.weight"].shape[1],
        num_classes=probe_state["fc.weight"].shape[0]
    ).to(DEVICE)
    linear_probe.load_state_dict(probe_state, strict=True)
    linear_probe.eval()

    return model, linear_probe, official_fisher_indices, num_prefix


# ============================================================
# 单图 System 1 预扫描
# ============================================================

@torch.no_grad()
def scan_one_image(
    img_path: str,
    gt_label: str,
    dataset_name: str,
    subset_name: str,
    model,
    linear_probe,
    official_fisher_indices,
    num_prefix: int,
    score_th: float,
    ratio_th: float,
    token_th: float,
) -> Dict[str, Any]:
    raw_img = Image.open(img_path).convert("RGB")
    input_tensor = TRANSFORM(raw_img).unsqueeze(0).to(DEVICE)
    gray_img = tensor_to_gray_uint8(input_tensor)

    outputs = model(input_tensor, return_protocol=True)
    token_sequence = outputs["token_sequence"]

    # 全局分类概率
    selected_tokens = token_sequence[:, official_fisher_indices, :]
    probe_logits = linear_probe(selected_tokens)
    probs = torch.softmax(probe_logits, dim=-1)[0]
    fake_prob = probs[FAKE_CLASS_IDX].item()

    # patch 级 suspiciousness
    patch_tokens = token_sequence[:, num_prefix:, :]
    w_real = linear_probe.fc.weight[REAL_CLASS_IDX]
    w_fake = linear_probe.fc.weight[FAKE_CLASS_IDX]
    delta_w = torch.nn.functional.normalize(w_fake - w_real, dim=0)
    patch_norm = torch.nn.functional.normalize(patch_tokens[0], dim=-1)
    forensic_scores = torch.matmul(patch_norm, delta_w)

    topk = min(10, forensic_scores.shape[0])
    _, top_k_indices_patch = torch.topk(forensic_scores, k=topk)

    runtime_top_k_abs = (top_k_indices_patch + num_prefix).cpu().tolist()
    clusters = cluster_patch_indices(runtime_top_k_abs, num_prefix=num_prefix)

    proposal_objects = []
    num_singletons = 0

    for cluster in clusters:
        if len(cluster) < 2:
            num_singletons += 1
            continue

        bbox = cluster_to_bbox(cluster)
        _, contrast_ratio, _, _ = validate_region_with_neighbor(gray_img, bbox)

        source_indices = [x[0] for x in cluster]
        cluster_scores = [forensic_scores[idx - num_prefix].item() for idx in source_indices]
        score_max = max(cluster_scores)

        # token contrast：取 cluster 内 patch score 的范围
        token_contrast = max(cluster_scores) - min(cluster_scores) if len(cluster_scores) >= 2 else 0.0

        proposal_objects.append({
            "score_max": score_max,
            "contrast_ratio": contrast_ratio,
            "token_contrast": token_contrast,
            "num_patches": len(cluster),
        })

    best_score_max = max([o["score_max"] for o in proposal_objects]) if proposal_objects else 0.0
    best_contrast_ratio = max([o["contrast_ratio"] for o in proposal_objects]) if proposal_objects else 0.0
    best_token_contrast = max([o["token_contrast"] for o in proposal_objects]) if proposal_objects else 0.0

    pixel_any_passed = best_score_max > score_th
    token_any_passed = best_token_contrast > token_th
    has_any_passed = pixel_any_passed or token_any_passed
    has_high_conf = pixel_any_passed and (best_contrast_ratio > ratio_th)

    if has_high_conf:
        status = "HighConfidenceEvidence"
    elif has_any_passed:
        status = "WeakEvidence"
    else:
        status = "NoEvidence"

    # evidence coverage：简单定义为通过 score 阈值的 proposal patch 总数 / topk patch 数
    proposal_patch_sum = sum([o["num_patches"] for o in proposal_objects])
    evidence_coverage = float(proposal_patch_sum / max(topk, 1))

    row = {
        "img_path": img_path,
        "filename": os.path.basename(img_path),
        "dataset_name": dataset_name,
        "subset_name": subset_name,
        "gt_label": gt_label,

        "status": status,
        "status_high": int(status == "HighConfidenceEvidence"),
        "status_weak": int(status == "WeakEvidence"),
        "status_none": int(status == "NoEvidence"),

        "fake_prob": float(fake_prob),
        "best_score_max": float(best_score_max),
        "best_contrast_ratio": float(best_contrast_ratio),
        "best_token_contrast": float(best_token_contrast),
        "evidence_coverage": float(evidence_coverage),

        "num_raw_clusters": int(len(clusters)),
        "num_singletons": int(num_singletons),
        "num_valid_clusters": int(len(proposal_objects)),

        "pixel_any_passed": int(pixel_any_passed),
        "token_any_passed": int(token_any_passed),
        "has_any_passed": int(has_any_passed),
        "has_high_conf": int(has_high_conf),
    }
    return row


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="System 1 预扫描 + 自动生成总表")
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--max_per_leaf_per_label", type=int, default=DEFAULT_MAX_PER_LEAF_PER_LABEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--score_th", type=float, default=DEFAULT_SCORE_TH)
    parser.add_argument("--ratio_th", type=float, default=DEFAULT_RATIO_TH)
    parser.add_argument("--token_th", type=float, default=DEFAULT_TOKEN_TH)
    args = parser.parse_args()

    set_seed(args.seed)

    model, linear_probe, official_fisher_indices, num_prefix = load_system1()

    all_leaf_groups = []
    for root in DATASET_ROOTS:
        if not os.path.isdir(root):
            print(f"[!] 跳过不存在目录: {root}")
            continue
        groups = find_label_dirs(root)
        if len(groups) == 0:
            print(f"[!] 未发现 0_real / 1_fake 叶子目录: {root}")
            continue
        all_leaf_groups.extend(groups)

    print(f"[*] 共发现 {len(all_leaf_groups)} 个可扫描叶子组")

    rows = []

    for group_idx, group in enumerate(all_leaf_groups, start=1):
        dataset_root = group["dataset_root"]
        subset_name = group["subset_name"]
        dataset_name = os.path.basename(dataset_root)

        real_paths = list_images_in_dir(group["real_dir"])
        fake_paths = list_images_in_dir(group["fake_dir"])

        real_paths = sample_paths(real_paths, args.max_per_leaf_per_label, args.seed + group_idx)
        fake_paths = sample_paths(fake_paths, args.max_per_leaf_per_label, args.seed + group_idx + 999)

        print("=" * 100)
        print(f"[*] [{group_idx}/{len(all_leaf_groups)}] 扫描 {subset_name}")
        print(f"    Real: {len(real_paths)} 张 | Fake: {len(fake_paths)} 张")

        for img_path in real_paths:
            try:
                row = scan_one_image(
                    img_path=img_path,
                    gt_label="Real",
                    dataset_name=dataset_name,
                    subset_name=subset_name,
                    model=model,
                    linear_probe=linear_probe,
                    official_fisher_indices=official_fisher_indices,
                    num_prefix=num_prefix,
                    score_th=args.score_th,
                    ratio_th=args.ratio_th,
                    token_th=args.token_th,
                )
                rows.append(row)
            except Exception as e:
                print(f"[!] 跳过 {img_path}: {e}")

        for img_path in fake_paths:
            try:
                row = scan_one_image(
                    img_path=img_path,
                    gt_label="Fake",
                    dataset_name=dataset_name,
                    subset_name=subset_name,
                    model=model,
                    linear_probe=linear_probe,
                    official_fisher_indices=official_fisher_indices,
                    num_prefix=num_prefix,
                    score_th=args.score_th,
                    ratio_th=args.ratio_th,
                    token_th=args.token_th,
                )
                rows.append(row)
            except Exception as e:
                print(f"[!] 跳过 {img_path}: {e}")

    if len(rows) == 0:
        raise RuntimeError("[!] 没有成功扫描任何图片。")

    fieldnames = [
        "img_path",
        "filename",
        "dataset_name",
        "subset_name",
        "gt_label",

        "status",
        "status_high",
        "status_weak",
        "status_none",

        "fake_prob",
        "best_score_max",
        "best_contrast_ratio",
        "best_token_contrast",
        "evidence_coverage",

        "num_raw_clusters",
        "num_singletons",
        "num_valid_clusters",

        "pixel_any_passed",
        "token_any_passed",
        "has_any_passed",
        "has_high_conf",
    ]

    ensure_parent_dir(args.output_csv)
    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print(f"[+] 预扫描完成，总样本数: {len(rows)}")
    print(f"[+] 总表已保存到: {args.output_csv}")

    # 简单汇总
    bucket_counter = {}
    for r in rows:
        key = (r["gt_label"], r["status"])
        bucket_counter[key] = bucket_counter.get(key, 0) + 1

    print("[+] bucket 分布：")
    for key in sorted(bucket_counter.keys()):
        print(f"    {key}: {bucket_counter[key]}")


if __name__ == "__main__":
    main()