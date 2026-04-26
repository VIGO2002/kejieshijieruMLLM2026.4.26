import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List

import torch

# 直接复用你已有的导出函数
from export_system2_cases import load_system1, export_case


def read_csv(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_label(x: str) -> str:
    s = str(x).strip().lower()
    if s == "real":
        return "Real"
    if s == "fake":
        return "Fake"
    raise ValueError(f"[!] 未知 gt_label: {x}")


def build_case_cfgs(rows: List[Dict], split_name: str) -> List[Dict]:
    case_cfgs = []

    for row in rows:
        case_name = row["case_name"]
        gt_label = normalize_label(row["gt_label"])
        image_path = row["image_path"]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"[!] 找不到图片: {image_path}")

        filename = os.path.basename(image_path)
        stem = Path(filename).stem

        # 这里的 case_tag 只影响导出目录名，不影响 System1 推理逻辑
        case_tag = f"TinyGenImage__{split_name}__{gt_label}"

        case_cfgs.append({
            "case_tag": case_tag,
            "filename": filename,
            "dataset_name": "Tiny-GenImage",
            "image_path": image_path,
            "label": gt_label,
            "manifest_case_name": case_name,
            "relative_path": row.get("relative_path", ""),
            "source_type": row.get("source_type", ""),
            "split": row.get("split", split_name),
        })

    return case_cfgs


def main():
    parser = argparse.ArgumentParser(description="Export Tiny-GenImage cases using existing System1 export pipeline")
    parser.add_argument(
        "--manifest_csv",
        type=str,
        required=True,
        help="Manifest built by build_tiny_genimage_manifest.py"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="tiny_genimage_eval/cases_validation",
        help="Output root for exported case folders"
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="validation",
        help="Split name used in case_tag"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only export first N samples; 0 means all"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip case if export dir already exists and contains evidence json"
    )
    args = parser.parse_args()

    rows = read_csv(args.manifest_csv)
    if args.limit > 0:
        rows = rows[:args.limit]

    case_cfgs = build_case_cfgs(rows, args.split_name)

    num_real = sum(1 for x in case_cfgs if x["label"] == "Real")
    num_fake = sum(1 for x in case_cfgs if x["label"] == "Fake")

    print(f"[*] 从 manifest 读取样本数: {len(case_cfgs)}")
    print(f"    Real: {num_real}")
    print(f"    Fake: {num_fake}")
    print(f"[*] 输出目录: {args.output_root}")

    os.makedirs(args.output_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 正在初始化 System 1 导出环境 ({device}) ...")
    system1_bundle = load_system1(device)

    success = []
    skipped = []
    failed = []

    for idx, cfg in enumerate(case_cfgs, start=1):
        filename = cfg["filename"]
        stem = Path(filename).stem
        expected_dir = os.path.join(args.output_root, f"{cfg['case_tag']}__{stem}")

        print(f"[*] [{idx}/{len(case_cfgs)}] exporting: {cfg['manifest_case_name']}")

        if args.skip_existing and os.path.isdir(expected_dir):
            has_evidence = any(
                os.path.exists(os.path.join(expected_dir, f"{stem}_{suffix}"))
                for suffix in ["evidence_full.json", "evidence_for_model.json"]
            )
            if has_evidence:
                skipped.append({
                    "manifest_case_name": cfg["manifest_case_name"],
                    "case_dir": expected_dir,
                    "reason": "already_exported"
                })
                print("    -> skipped (already exported)")
                continue

        try:
            export_case(
                case_cfg=cfg,
                export_root=args.output_root,
                system1_bundle=system1_bundle,
                device=device
            )
            success.append({
                "manifest_case_name": cfg["manifest_case_name"],
                "case_dir": expected_dir,
                "label": cfg["label"]
            })
        except Exception as e:
            failed.append({
                "manifest_case_name": cfg["manifest_case_name"],
                "image_path": cfg["image_path"],
                "label": cfg["label"],
                "error": repr(e)
            })
            print(f"    -> failed: {e}")

    summary = {
        "manifest_csv": args.manifest_csv,
        "output_root": args.output_root,
        "num_total": len(case_cfgs),
        "num_success": len(success),
        "num_skipped": len(skipped),
        "num_failed": len(failed),
        "split_name": args.split_name,
    }

    write_json(os.path.join(args.output_root, "summary.json"), summary)

    if success:
        import pandas as pd
        pd.DataFrame(success).to_csv(
            os.path.join(args.output_root, "success_cases.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    if skipped:
        import pandas as pd
        pd.DataFrame(skipped).to_csv(
            os.path.join(args.output_root, "skipped_cases.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    if failed:
        import pandas as pd
        pd.DataFrame(failed).to_csv(
            os.path.join(args.output_root, "failed_cases.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    print("\n[+] Tiny-GenImage 导出完成")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()