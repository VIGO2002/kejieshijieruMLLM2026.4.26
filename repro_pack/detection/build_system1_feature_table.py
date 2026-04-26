"""
build_system1_feature_table.py

功能：
1. 遍历 system2_case_exports_18 / system2_holdout_exports 这类导出目录
2. 从 *_evidence_full.json 中提取已有 System 1 连续证据
3. 使用冻结的 DINOv3 提取免训练全局特征：
   - prefix_mean_feature（前缀 token 均值）
   - patch_gap_feature（patch token 全局平均）
4. 在 reference_real_bank 上计算真实图像中心
5. 计算每个样本到真实中心的余弦距离：
   - sys1_global_anomaly_prefix
   - sys1_global_anomaly_gap
6. 输出 system1_feature_table.csv，供后续 NoEvidence expert / learned fusion 使用
"""

import os
import glob
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from test_system1_protocol import DinoV3Model


# ============================================================
# 默认配置
# ============================================================

DEFAULT_EXPORTS_DIR = "system2_holdout_exports"   # 也可以改成 system2_case_exports_18
DEFAULT_REFERENCE_REAL_DIR = "reference_real_bank"
DEFAULT_OUTPUT_CSV = "system1_feature_table.csv"
DEFAULT_CENTERS_JSON = "real_feature_centers.json"

DEFAULT_MODEL_NAME = "dinov3_vit_7b"
DEFAULT_POOL_TYPE = "patch_avg"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# ============================================================
# 工具函数
# ============================================================

def ensure_dir_for_file(path: str):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Dict[str, Any]):
    ensure_dir_for_file(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def find_case_jsons(exports_dir: str) -> List[str]:
    # 对齐你当前导出结构：EXPORTS_DIR / case_name / *_evidence_full.json
    return sorted(glob.glob(os.path.join(exports_dir, "*", "*_evidence_full.json")))


def find_original_raw(case_dir: str) -> str:
    # 对齐你当前导出结构：同目录下有 *_original_raw.*
    for p in Path(case_dir).iterdir():
        if p.is_file() and "_original_raw" in p.name:
            return str(p)
    raise FileNotFoundError(f"[!] 未找到 original_raw: {case_dir}")


def list_reference_real_images(reference_real_dir: str) -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG", "*.webp", "*.WEBP"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(reference_real_dir, ext)))
    return sorted(set(files))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def mean_topk(values: List[float], k: int) -> float:
    if not values:
        return 0.0
    vals = sorted(values, reverse=True)[:k]
    return float(sum(vals) / len(vals))


def max_or_zero(values: List[float]) -> float:
    return float(max(values)) if values else 0.0


def sum_int(values: List[int]) -> int:
    return int(sum(values)) if values else 0


def max_int_or_zero(values: List[int]) -> int:
    return int(max(values)) if values else 0


# ============================================================
# DINOv3 全局特征提取
# ============================================================

@torch.no_grad()
def extract_global_features(
    model: DinoV3Model,
    img_tensor: torch.Tensor,
    num_prefix: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回：
    - prefix_mean_feature: [1, D]
    - patch_gap_feature  : [1, D]
    """
    img_tensor = img_tensor.to(DEVICE)
    outputs = model(img_tensor, return_protocol=True)
    token_sequence = outputs["token_sequence"]   # [B, N, D]

    prefix_tokens = token_sequence[:, :num_prefix, :]
    patch_tokens = token_sequence[:, num_prefix:, :]

    prefix_mean_feature = prefix_tokens.mean(dim=1)   # [B, D]
    patch_gap_feature = patch_tokens.mean(dim=1)      # [B, D]

    return prefix_mean_feature, patch_gap_feature


@torch.no_grad()
def extract_image_global_features(
    model: DinoV3Model,
    image_path: str,
    num_prefix: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)
    p_mean, p_gap = extract_global_features(model, tensor, num_prefix)

    # 先归一化，再用于中心计算 / 距离计算
    p_mean = F.normalize(p_mean, dim=-1)
    p_gap = F.normalize(p_gap, dim=-1)
    return p_mean, p_gap


def compute_real_centers(
    model: DinoV3Model,
    reference_real_dir: str,
    centers_json: str,
    num_prefix: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    print(f"[*] 正在计算真实图像中心: {reference_real_dir}")
    ref_paths = list_reference_real_images(reference_real_dir)

    if len(ref_paths) == 0:
        raise FileNotFoundError(
            f"[!] 参考真图库为空: {reference_real_dir}\n"
            f"请先准备 reference_real_bank，并放入 50-100 张风格尽量多样的真实图片。"
        )

    prefix_feats = []
    gap_feats = []

    for path in tqdm(ref_paths, desc="提取参考真图全局特征"):
        try:
            p_mean, p_gap = extract_image_global_features(model, path, num_prefix)
            prefix_feats.append(p_mean.cpu())
            gap_feats.append(p_gap.cpu())
        except Exception as e:
            print(f"[!] 跳过参考图 {path}: {e}")

    if len(prefix_feats) == 0 or len(gap_feats) == 0:
        raise RuntimeError("[!] 未能成功提取任何参考真图特征。")

    center_prefix = torch.cat(prefix_feats, dim=0).mean(dim=0, keepdim=True)
    center_gap = torch.cat(gap_feats, dim=0).mean(dim=0, keepdim=True)

    center_prefix = F.normalize(center_prefix, dim=-1)
    center_gap = F.normalize(center_gap, dim=-1)

    centers_payload = {
        "reference_real_dir": reference_real_dir,
        "num_reference_images": len(prefix_feats),
        "feature_dim": int(center_prefix.shape[-1]),
        "center_prefix": center_prefix.cpu().tolist(),
        "center_gap": center_gap.cpu().tolist(),
    }
    save_json(centers_json, centers_payload)

    print(f"[+] 已保存真实图像中心到: {centers_json}")
    return center_prefix.to(DEVICE), center_gap.to(DEVICE)


def load_or_compute_real_centers(
    model: DinoV3Model,
    reference_real_dir: str,
    centers_json: str,
    num_prefix: int,
    force_recompute: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (not force_recompute) and os.path.exists(centers_json):
        print(f"[*] 发现已有中心文件，直接加载: {centers_json}")
        data = load_json(centers_json)
        center_prefix = torch.tensor(data["center_prefix"], dtype=torch.float32, device=DEVICE)
        center_gap = torch.tensor(data["center_gap"], dtype=torch.float32, device=DEVICE)
        center_prefix = F.normalize(center_prefix, dim=-1)
        center_gap = F.normalize(center_gap, dim=-1)
        return center_prefix, center_gap

    return compute_real_centers(model, reference_real_dir, centers_json, num_prefix)


# ============================================================
# 从 evidence_full.json 派生更多统计特征
# ============================================================

def derive_object_features(data: Dict[str, Any]) -> Dict[str, Any]:
    proposal_objects = data.get("proposal_objects", []) or []
    validated_objects = data.get("validated_evidence_objects", []) or []

    proposal_scores = [safe_float(obj.get("score_max", 0.0)) for obj in proposal_objects]
    proposal_token = [safe_float(obj.get("token_contrast", 0.0)) for obj in proposal_objects]
    proposal_ratio = [safe_float(obj.get("contrast_ratio", 0.0)) for obj in proposal_objects]
    proposal_np = [safe_int(obj.get("num_patches", 0)) for obj in proposal_objects]

    validated_scores = [safe_float(obj.get("score_max", 0.0)) for obj in validated_objects]
    validated_token = [safe_float(obj.get("token_contrast", 0.0)) for obj in validated_objects]
    validated_ratio = [safe_float(obj.get("contrast_ratio", 0.0)) for obj in validated_objects]
    validated_np = [safe_int(obj.get("num_patches", 0)) for obj in validated_objects]

    num_high = 0
    num_weak = 0
    num_rejected = 0
    for obj in proposal_objects:
        level = str(obj.get("evidence_level", "Rejected"))
        if level == "HighConfidenceEvidence":
            num_high += 1
        elif level == "WeakEvidence":
            num_weak += 1
        else:
            num_rejected += 1

    feats = {
        "num_proposal_objects": len(proposal_objects),
        "num_validated_objects": len(validated_objects),

        "num_high_objects": num_high,
        "num_weak_objects": num_weak,
        "num_rejected_objects": num_rejected,

        "proposal_score_top1": max_or_zero(proposal_scores),
        "proposal_score_top3_mean": mean_topk(proposal_scores, 3),
        "proposal_score_top5_mean": mean_topk(proposal_scores, 5),

        "proposal_token_top1": max_or_zero(proposal_token),
        "proposal_token_top3_mean": mean_topk(proposal_token, 3),
        "proposal_token_top5_mean": mean_topk(proposal_token, 5),

        "proposal_ratio_top1": max_or_zero(proposal_ratio),
        "proposal_ratio_top3_mean": mean_topk(proposal_ratio, 3),

        "proposal_patch_top1": max_int_or_zero(proposal_np),
        "proposal_patch_sum": sum_int(proposal_np),

        "validated_score_top1": max_or_zero(validated_scores),
        "validated_score_top3_mean": mean_topk(validated_scores, 3),

        "validated_token_top1": max_or_zero(validated_token),
        "validated_token_top3_mean": mean_topk(validated_token, 3),

        "validated_ratio_top1": max_or_zero(validated_ratio),
        "validated_ratio_top3_mean": mean_topk(validated_ratio, 3),

        "validated_patch_top1": max_int_or_zero(validated_np),
        "validated_patch_sum": sum_int(validated_np),
    }
    return feats


# ============================================================
# 单样本构造特征行
# ============================================================

def build_feature_row(
    case_dir: str,
    json_path: str,
    model: DinoV3Model,
    num_prefix: int,
    center_prefix: torch.Tensor,
    center_gap: torch.Tensor
) -> Optional[Dict[str, Any]]:
    data = load_json(json_path)

    case_name = Path(case_dir).name
    raw_image_path = find_original_raw(case_dir)

    # 读取已有 System 1 特征
    row = {
        "case_name": case_name,
        "case_tag": data.get("case_tag", ""),
        "dataset": data.get("dataset", ""),
        "source_type": data.get("source_type", ""),
        "filename": data.get("filename", ""),
        "gt_label": data.get("label", "Unknown"),
        "pred_label": data.get("pred_label", "Unknown"),
        "cls_correct": safe_int(data.get("cls_correct", 0)),

        "status": data.get("status", "NoEvidence"),
        "status_high": int(data.get("status", "NoEvidence") == "HighConfidenceEvidence"),
        "status_weak": int(data.get("status", "NoEvidence") == "WeakEvidence"),
        "status_none": int(data.get("status", "NoEvidence") == "NoEvidence"),

        "fake_prob": safe_float(data.get("fake_prob", 0.0)),
        "best_score_max": safe_float(data.get("best_score_max", 0.0)),
        "best_contrast_ratio": safe_float(data.get("best_contrast_ratio", 0.0)),
        "best_token_contrast": safe_float(data.get("best_token_contrast", 0.0)),
        "evidence_coverage": safe_float(data.get("evidence_coverage", 0.0)),

        "num_raw_clusters": safe_int(data.get("num_raw_clusters", 0)),
        "num_singletons": safe_int(data.get("num_singletons", 0)),
        "num_valid_clusters": safe_int(data.get("num_valid_clusters", 0)),

        "pixel_any_passed": int(bool(data.get("pixel_any_passed", False))),
        "token_any_passed": int(bool(data.get("token_any_passed", False))),
        "has_any_passed": int(bool(data.get("has_any_passed", False))),
        "has_high_conf": int(bool(data.get("has_high_conf", False))),
        "rescued_by_token": int(bool(data.get("rescued_by_token", False))),
    }

    # 派生 proposal/validated object 统计
    row.update(derive_object_features(data))

    # 提取免训练全局特征并计算 anomaly
    try:
        p_mean, p_gap = extract_image_global_features(model, raw_image_path, num_prefix)

        # 都已 L2 normalize
        dist_prefix = 1.0 - F.cosine_similarity(p_mean.to(DEVICE), center_prefix, dim=-1).item()
        dist_gap = 1.0 - F.cosine_similarity(p_gap.to(DEVICE), center_gap, dim=-1).item()

        row["sys1_global_anomaly_prefix"] = float(dist_prefix)
        row["sys1_global_anomaly_gap"] = float(dist_gap)
    except Exception as e:
        print(f"[!] 提取全局特征失败 {case_name}: {e}")
        row["sys1_global_anomaly_prefix"] = 0.0
        row["sys1_global_anomaly_gap"] = 0.0

    return row


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports_dir", type=str, default=DEFAULT_EXPORTS_DIR)
    parser.add_argument("--reference_real_dir", type=str, default=DEFAULT_REFERENCE_REAL_DIR)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--centers_json", type=str, default=DEFAULT_CENTERS_JSON)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pool_type", type=str, default=DEFAULT_POOL_TYPE)
    parser.add_argument("--force_recompute_centers", action="store_true")
    args = parser.parse_args()

    print(f"[*] 初始化 DINOv3 ({DEVICE}) ...")
    model = DinoV3Model(model_name=args.model_name, pool_type=args.pool_type).to(DEVICE)
    model.eval()
    num_prefix = getattr(model.backbone, "num_prefix_tokens", 5)

    center_prefix, center_gap = load_or_compute_real_centers(
        model=model,
        reference_real_dir=args.reference_real_dir,
        centers_json=args.centers_json,
        num_prefix=num_prefix,
        force_recompute=args.force_recompute_centers
    )

    json_files = find_case_jsons(args.exports_dir)
    print(f"[*] 在 {args.exports_dir} 中找到 {len(json_files)} 个 *_evidence_full.json")

    if len(json_files) == 0:
        raise FileNotFoundError(
            f"[!] 没找到任何 *_evidence_full.json。\n"
            f"请检查 exports_dir 是否正确，例如：system2_case_exports_18 或 system2_holdout_exports"
        )

    rows = []
    for jpath in tqdm(json_files, desc="构建 System 1 特征表"):
        case_dir = str(Path(jpath).parent)
        try:
            row = build_feature_row(
                case_dir=case_dir,
                json_path=jpath,
                model=model,
                num_prefix=num_prefix,
                center_prefix=center_prefix,
                center_gap=center_gap
            )
            if row is not None:
                rows.append(row)
        except Exception as e:
            print(f"[!] 跳过 {jpath}: {e}")

    if len(rows) == 0:
        raise RuntimeError("[!] 没有成功构建任何特征行。")

    fieldnames = [
        "case_name", "case_tag", "dataset", "source_type", "filename",
        "gt_label", "pred_label", "cls_correct",
        "status", "status_high", "status_weak", "status_none",

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
        "rescued_by_token",

        "num_proposal_objects",
        "num_validated_objects",
        "num_high_objects",
        "num_weak_objects",
        "num_rejected_objects",

        "proposal_score_top1",
        "proposal_score_top3_mean",
        "proposal_score_top5_mean",
        "proposal_token_top1",
        "proposal_token_top3_mean",
        "proposal_token_top5_mean",
        "proposal_ratio_top1",
        "proposal_ratio_top3_mean",
        "proposal_patch_top1",
        "proposal_patch_sum",

        "validated_score_top1",
        "validated_score_top3_mean",
        "validated_token_top1",
        "validated_token_top3_mean",
        "validated_ratio_top1",
        "validated_ratio_top3_mean",
        "validated_patch_top1",
        "validated_patch_sum",

        "sys1_global_anomaly_prefix",
        "sys1_global_anomaly_gap",
    ]

    ensure_dir_for_file(args.output_csv)
    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] 已生成 System 1 特征表: {args.output_csv}")
    print(f"[+] 总样本数: {len(rows)}")
    print(f"[+] 真实中心文件: {args.centers_json}")


if __name__ == "__main__":
    main()