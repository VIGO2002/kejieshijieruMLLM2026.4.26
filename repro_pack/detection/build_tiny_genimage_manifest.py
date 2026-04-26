# build_tiny_genimage_manifest.py

import os
import csv
import argparse
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def safe_case_name(split: str, label_name: str, rel_path: Path) -> str:
    stem = rel_path.with_suffix("").as_posix()
    stem = stem.replace("/", "__").replace("\\", "__").replace(" ", "_")
    return f"TinyGenImage__{split}__{label_name}__{stem}"


def collect_images(label_dir: Path, split: str, gt_label: str):
    rows = []
    for p in sorted(label_dir.rglob("*")):
        if not is_image_file(p):
            continue
        rel_path = p.relative_to(label_dir)
        case_name = safe_case_name(split, gt_label, rel_path)
        rows.append({
            "case_name": case_name,
            "dataset": "tiny-genimage",
            "split": split,
            "gt_label": gt_label,
            "image_path": str(p.resolve()),
            "relative_path": rel_path.as_posix(),
            "source_type": label_dir.name,
        })
    return rows


def write_csv(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "case_name",
        "dataset",
        "split",
        "gt_label",
        "image_path",
        "relative_path",
        "source_type",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build manifest for Tiny-GenImage")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Tiny-GenImage root, e.g. /root/autodl-tmp/datasets/tiny-genimage"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Which split to scan"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="tiny_genimage_eval/manifest_validation.csv",
        help="Output manifest CSV path"
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    split_dir = root_dir / args.split
    real_dir = split_dir / "0_real"
    fake_dir = split_dir / "1_fake"

    if not real_dir.exists():
        raise FileNotFoundError(f"[!] real dir not found: {real_dir}")
    if not fake_dir.exists():
        raise FileNotFoundError(f"[!] fake dir not found: {fake_dir}")

    rows = []
    rows.extend(collect_images(real_dir, args.split, "Real"))
    rows.extend(collect_images(fake_dir, args.split, "Fake"))

    rows = sorted(rows, key=lambda x: (x["gt_label"], x["relative_path"]))

    write_csv(args.output_csv, rows)

    n_real = sum(1 for r in rows if r["gt_label"] == "Real")
    n_fake = sum(1 for r in rows if r["gt_label"] == "Fake")

    print("[+] Tiny-GenImage manifest built")
    print(f"[*] split       : {args.split}")
    print(f"[*] total       : {len(rows)}")
    print(f"[*] real        : {n_real}")
    print(f"[*] fake        : {n_fake}")
    print(f"[*] output_csv  : {args.output_csv}")


if __name__ == "__main__":
    main()