"""
select_holdout_from_remaining_scan.py

功能：
1. 从 system1_scan_table.csv 中读取全部预扫描结果
2. 读取 train_pool_selected/selected_manifest.csv，排除已进入 train pool 的原始样本
3. 从剩余样本中按 bucket 自动采样，构建独立 holdout
4. 复制图片到 holdout_pool_selected/
5. 保存 holdout_pool_selected/selected_manifest.csv

默认目标：
- Fake / NoEvidence: 25
- Real / NoEvidence: 25
- Fake / WeakEvidence: 12
- Fake / HighConfidenceEvidence: 8
"""

import os
import csv
import shutil
import random
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple


VALID_LABELS = {"Fake", "Real"}
VALID_STATUS = {"NoEvidence", "WeakEvidence", "HighConfidenceEvidence"}

DEFAULT_SCAN_CSV = "system1_scan_table_v1.csv"
DEFAULT_TRAIN_MANIFEST = "train_pool_selected/selected_manifest.csv"
DEFAULT_OUTPUT_DIR = "holdout_pool_selected"
DEFAULT_OUTPUT_MANIFEST = "holdout_pool_selected/selected_manifest.csv"
DEFAULT_SEED = 42

DEFAULT_TARGET_BUCKETS = {
    ("Fake", "NoEvidence"): 25,
    ("Real", "NoEvidence"): 25,
    ("Fake", "WeakEvidence"): 12,
    ("Fake", "HighConfidenceEvidence"): 8,
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_csv_rows(csv_path: str) -> List[Dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[!] 找不到 CSV: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def bucket_key(row: Dict) -> Tuple[str, str]:
    return (str(row.get("gt_label", "")).strip(), str(row.get("status", "")).strip())


def subgroup_key(row: Dict) -> str:
    dataset_name = str(row.get("dataset_name", "")).strip()
    subset_name = str(row.get("subset_name", "")).strip()
    return f"{dataset_name}__{subset_name}"


def print_bucket_summary(rows: List[Dict], title: str):
    counter = defaultdict(int)
    for row in rows:
        gt = str(row.get("gt_label", "")).strip()
        status = str(row.get("status", "")).strip()
        if gt in VALID_LABELS and status in VALID_STATUS:
            counter[(gt, status)] += 1

    print("=" * 90)
    print(title)
    print("=" * 90)
    for key in sorted(counter.keys()):
        print(f"{key}: {counter[key]}")
    print()


def load_train_img_paths(train_manifest_csv: str) -> set:
    """
    从 train manifest 中读取已经进入 train pool 的原始 img_path
    """
    rows = load_csv_rows(train_manifest_csv)
    used = set()
    for row in rows:
        img_path = str(row.get("img_path", "")).strip()
        if img_path:
            used.add(img_path)
    return used


def filter_remaining_rows(scan_rows: List[Dict], used_img_paths: set) -> List[Dict]:
    kept = []
    for row in scan_rows:
        img_path = str(row.get("img_path", "")).strip()
        gt = str(row.get("gt_label", "")).strip()
        status = str(row.get("status", "")).strip()

        if not img_path or (not os.path.exists(img_path)):
            continue
        if gt not in VALID_LABELS:
            continue
        if status not in VALID_STATUS:
            continue
        if img_path in used_img_paths:
            continue

        kept.append(row)
    return kept


def group_by_bucket_and_subset(rows: List[Dict]) -> Dict[Tuple[str, str], Dict[str, List[Dict]]]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[bucket_key(row)][subgroup_key(row)].append(row)
    return grouped


def round_robin_select(subset_to_rows: Dict[str, List[Dict]], target_n: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)

    pools = {}
    for subset, rows in subset_to_rows.items():
        rows_copy = rows[:]
        rng.shuffle(rows_copy)
        pools[subset] = rows_copy

    subset_names = sorted(pools.keys())
    selected = []

    while len(selected) < target_n:
        progressed = False
        for subset in subset_names:
            if len(selected) >= target_n:
                break
            if pools[subset]:
                selected.append(pools[subset].pop(0))
                progressed = True
        if not progressed:
            break

    return selected


def copy_selected_rows(selected_rows: List[Dict], output_dir: str) -> List[Dict]:
    manifest_rows = []
    used_names = set()

    for idx, row in enumerate(selected_rows):
        gt = str(row["gt_label"]).strip()
        status = str(row["status"]).strip()
        dataset_name = str(row.get("dataset_name", "unknown")).strip()
        subset_name = str(row.get("subset_name", "unknown")).strip().replace("/", "__")
        src_path = str(row["img_path"]).strip()
        src_name = os.path.basename(src_path)

        bucket_dir = os.path.join(output_dir, f"{gt}_{status}")
        ensure_dir(bucket_dir)

        dst_name = f"{dataset_name}__{subset_name}__{idx:05d}__{src_name}"
        while dst_name in used_names:
            idx += 1
            dst_name = f"{dataset_name}__{subset_name}__{idx:05d}__{src_name}"
        used_names.add(dst_name)

        dst_path = os.path.join(bucket_dir, dst_name)
        shutil.copy2(src_path, dst_path)

        out_row = dict(row)
        out_row["selected_bucket"] = f"{gt}_{status}"
        out_row["copied_path"] = dst_path
        manifest_rows.append(out_row)

    return manifest_rows


def save_manifest(rows: List[Dict], manifest_path: str):
    if not rows:
        print("[!] 没有样本被选中，manifest 不会写出。")
        return

    fieldnames = list(rows[0].keys())
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        ensure_dir(manifest_dir)

    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="从剩余扫描表自动构建独立 holdout")
    parser.add_argument("--scan_csv", type=str, default=DEFAULT_SCAN_CSV)
    parser.add_argument("--train_manifest", type=str, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=str, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--n_fake_none", type=int, default=DEFAULT_TARGET_BUCKETS[("Fake", "NoEvidence")])
    parser.add_argument("--n_real_none", type=int, default=DEFAULT_TARGET_BUCKETS[("Real", "NoEvidence")])
    parser.add_argument("--n_fake_weak", type=int, default=DEFAULT_TARGET_BUCKETS[("Fake", "WeakEvidence")])
    parser.add_argument("--n_fake_high", type=int, default=DEFAULT_TARGET_BUCKETS[("Fake", "HighConfidenceEvidence")])

    parser.add_argument("--dry_run", action="store_true", help="只统计，不复制图片")
    args = parser.parse_args()

    target_buckets = {
        ("Fake", "NoEvidence"): args.n_fake_none,
        ("Real", "NoEvidence"): args.n_real_none,
        ("Fake", "WeakEvidence"): args.n_fake_weak,
        ("Fake", "HighConfidenceEvidence"): args.n_fake_high,
    }

    random.seed(args.seed)

    scan_rows = load_csv_rows(args.scan_csv)
    print(f"[*] 扫描总表样本数: {len(scan_rows)}")
    print_bucket_summary(scan_rows, "原始扫描表 bucket 分布")

    used_img_paths = load_train_img_paths(args.train_manifest)
    print(f"[*] train manifest 中已占用原始图片数: {len(used_img_paths)}")

    remaining_rows = filter_remaining_rows(scan_rows, used_img_paths)
    print(f"[*] 排除 train pool 后剩余有效样本数: {len(remaining_rows)}")
    print_bucket_summary(remaining_rows, "剩余可用于 holdout 的 bucket 分布")

    grouped = group_by_bucket_and_subset(remaining_rows)

    selected_rows = []
    print("[*] 开始构建 holdout ...")
    for bucket, target_n in target_buckets.items():
        subset_to_rows = grouped.get(bucket, {})
        available_n = sum(len(v) for v in subset_to_rows.values())

        print(f"    - {bucket}: 目标={target_n}, 可用={available_n}, 子来源数={len(subset_to_rows)}")
        if available_n == 0:
            print("      [!] 该 bucket 无可用样本，跳过。")
            continue

        chosen = round_robin_select(
            subset_to_rows=subset_to_rows,
            target_n=min(target_n, available_n),
            seed=args.seed + hash(bucket) % 10000
        )
        selected_rows.extend(chosen)

    print_bucket_summary(selected_rows, "已选中 holdout bucket 分布")

    if args.dry_run:
        print("[*] dry_run 模式，不复制图片。")
        return

    ensure_dir(args.output_dir)
    manifest_rows = copy_selected_rows(selected_rows, args.output_dir)
    save_manifest(manifest_rows, args.manifest)

    print(f"[+] 已复制 holdout 样本到: {args.output_dir}")
    print(f"[+] holdout manifest 已保存到: {args.manifest}")

    subset_counter = defaultdict(int)
    for row in manifest_rows:
        key = (
            row.get("dataset_name", ""),
            row.get("subset_name", ""),
            row.get("gt_label", ""),
            row.get("status", "")
        )
        subset_counter[key] += 1

    print("=" * 90)
    print("holdout 样本来源分布（dataset, subset, label, status）")
    print("=" * 90)
    for key in sorted(subset_counter.keys()):
        print(f"{key}: {subset_counter[key]}")


if __name__ == "__main__":
    main()