"""
export_train_pool_from_manifest.py

功能：
1. 读取 select_train_pool_from_scan.py 生成的 selected_manifest.csv
2. 为每个样本自动构造 case_cfg
3. 调用你现有的 export_system2_cases.py 中的 load_system1 / export_case
4. 生成完整导出包：
   - *_original_raw
   - *_original_224
   - *_overlay_raw
   - *_overlay_224
   - *_evidence_for_model.json
   - *_evidence_full.json
   - crops_raw/

说明：
- 要先运行 select_train_pool_from_scan.py 的正式版（非 dry_run）
- 默认优先使用 copied_path；如果没有，则退回 img_path
"""

import os
import csv
import argparse
from pathlib import Path

import torch

# 直接复用你已有的完整导出逻辑
from export_system2_cases import load_system1, export_case


# ============================================================
# 默认配置
# ============================================================

DEFAULT_MANIFEST = "train_pool_selected/selected_manifest.csv"
DEFAULT_EXPORT_ROOT = "system2_train_pool_exports"


# ============================================================
# 工具函数
# ============================================================

def infer_case_tag(gt_label: str, status: str) -> str:
    """
    把 manifest 中的标签组合成可读 case_tag
    """
    gt_label = str(gt_label).strip()
    status = str(status).strip()

    short_status_map = {
        "NoEvidence": "None",
        "WeakEvidence": "Weak",
        "HighConfidenceEvidence": "High",
    }
    short_status = short_status_map.get(status, status)

    return f"TrainPool__{gt_label}_{short_status}"


def resolve_image_path(row: dict) -> str:
    """
    优先用 copied_path（如果你已经复制到 train_pool_selected）
    否则回退到原始 img_path
    """
    copied_path = str(row.get("copied_path", "")).strip()
    img_path = str(row.get("img_path", "")).strip()

    if copied_path and os.path.exists(copied_path):
        return copied_path
    if img_path and os.path.exists(img_path):
        return img_path

    raise FileNotFoundError(
        f"[!] 找不到图片路径。\n"
        f"copied_path={copied_path}\n"
        f"img_path={img_path}"
    )


def build_case_cfg(row: dict) -> dict:
    gt_label = str(row.get("gt_label", "Unknown")).strip()
    status = str(row.get("status", "Unknown")).strip()
    dataset_name = str(row.get("dataset_name", "unknown")).strip()
    subset_name = str(row.get("subset_name", "unknown")).strip()
    filename = str(row.get("filename", "")).strip()

    image_path = resolve_image_path(row)
    case_tag = infer_case_tag(gt_label, status)

    # 这里 dataset_name 故意带上 subset，方便后续分析来源
    dataset_full_name = f"{dataset_name}/{subset_name}" if subset_name else dataset_name

    return {
        "case_tag": case_tag,
        "filename": filename if filename else os.path.basename(image_path),
        "dataset_name": dataset_full_name,
        "image_path": image_path,
        "label": gt_label,
    }


def load_manifest_rows(manifest_path: str):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"[!] 找不到 manifest: {manifest_path}")

    rows = []
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="根据 selected_manifest.csv 自动生成完整导出包")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--export_root", type=str, default=DEFAULT_EXPORT_ROOT)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 正在初始化 System 1 导出环境 ({device})...")

    rows = load_manifest_rows(args.manifest)
    print(f"[*] 读取 manifest 行数: {len(rows)}")

    if len(rows) == 0:
        raise RuntimeError("[!] manifest 为空，没有可导出的样本。")

    os.makedirs(args.export_root, exist_ok=True)
    system1_bundle = load_system1(device)

    exported = 0
    skipped = 0

    for idx, row in enumerate(rows, start=1):
        try:
            case_cfg = build_case_cfg(row)

            print("-" * 100)
            print(f"[*] [{idx}/{len(rows)}] 正在导出: {case_cfg['filename']}")
            print(f"    case_tag     : {case_cfg['case_tag']}")
            print(f"    dataset_name : {case_cfg['dataset_name']}")
            print(f"    label        : {case_cfg['label']}")
            print(f"    image_path   : {case_cfg['image_path']}")

            export_case(
                case_cfg=case_cfg,
                export_root=args.export_root,
                system1_bundle=system1_bundle,
                device=device
            )
            exported += 1

        except Exception as e:
            skipped += 1
            print(f"[!] 跳过第 {idx} 个样本: {e}")

    print("=" * 100)
    print(f"[+] 导出完成")
    print(f"    成功导出: {exported}")
    print(f"    跳过数量: {skipped}")
    print(f"    导出目录: {args.export_root}")


if __name__ == "__main__":
    main()