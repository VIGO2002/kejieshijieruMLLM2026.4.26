"""
select_train_pool_from_scan.py

功能：
1. 从 system1_scan_table.csv 自动按 bucket 挑选目标样本
2. 尽量保持 dataset/subset 多样性（round-robin 轮转采样）
3. 将图片复制到 train_pool_selected/
4. 生成 selected_manifest.csv，供后续完整导出使用

默认策略（适合你当前阶段）：
- 重点补 NoEvidence Fake / NoEvidence Real
- 同时补一部分 WeakEvidence Fake
- 少量保留 HighConfidenceEvidence Fake
"""

import os
import csv
import shutil
import random
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple


# ============================================================
# 默认配置
# ============================================================

DEFAULT_INPUT_CSV = "system1_scan_table_v1.csv"
DEFAULT_OUTPUT_DIR = "train_pool_selected"
DEFAULT_MANIFEST = "train_pool_selected/selected_manifest.csv"
DEFAULT_SEED = 42

# 当前阶段推荐默认目标
DEFAULT_TARGET_BUCKETS = {
    ("Fake", "NoEvidence"): 40,
    ("Real", "NoEvidence"): 40,
    ("Fake", "WeakEvidence"): 20,
    ("Fake", "HighConfidenceEvidence"): 10,
    # 如果你后面想加，可以手动打开
    # ("Real", "WeakEvidence"): 2,
    # ("Real", "HighConfidenceEvidence"): 2,
}

VALID_LABELS = {"Fake", "Real"}
VALID_STATUS = {"NoEvidence", "WeakEvidence", "HighConfidenceEvidence"}


# ============================================================
# 工具函数
# ============================================================

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_scan_rows(csv_path: str) -> List[Dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[!] 找不到扫描总表: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = str(row.get("gt_label", "")).strip()
            status = str(row.get("status", "")).strip()
            img_path = str(row.get("img_path", "")).strip()

            if gt not in VALID_LABELS:
                continue
            if status not in VALID_STATUS:
                continue
            if not img_path or (not os.path.exists(img_path)):
                continue

            rows.append(row)
    return rows


def bucket_key(row: Dict) -> Tuple[str, str]:
    return (row["gt_label"], row["status"])


def subgroup_key(row: Dict) -> str:
    # 用 dataset + subset 控制多样性
    dataset_name = str(row.get("dataset_name", "")).strip()
    subset_name = str(row.get("subset_name", "")).strip()
    return f"{dataset_name}__{subset_name}"


def group_by_bucket_and_subset(rows: List[Dict]) -> Dict[Tuple[str, str], Dict[str, List[Dict]]]:
    """
    返回：
    {
      (gt_label, status): {
          "dataset__subset": [row1, row2, ...]
      }
    }
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[bucket_key(row)][subgroup_key(row)].append(row)
    return grouped


def round_robin_select(
    subset_to_rows: Dict[str, List[Dict]],
    target_n: int,
    seed: int
) -> List[Dict]:
    """
    尽量跨 subset 均衡抽样
    """
    rng = random.Random(seed)

    # 每个 subset 内部先打乱
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
            if len(pools[subset]) > 0:
                selected.append(pools[subset].pop(0))
                progressed = True
        if not progressed:
            break

    return selected


def copy_selected_rows(selected_rows: List[Dict], output_dir: str) -> List[Dict]:
    """
    复制图片并返回带目标路径的 manifest rows
    """
    manifest_rows = []
    used_names = set()

    for idx, row in enumerate(selected_rows):
        gt = row["gt_label"]
        status = row["status"]
        dataset_name = str(row.get("dataset_name", "unknown")).strip()
        subset_name = str(row.get("subset_name", "unknown")).strip().replace("/", "__")
        src_path = row["img_path"]
        src_name = os.path.basename(src_path)

        bucket_dir = os.path.join(output_dir, f"{gt}_{status}")
        ensure_dir(bucket_dir)

        # 防止重名：前面加 dataset/subset/index
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
        print("[!] 没有任何样本被选中，manifest 不会写出。")
        return

    fieldnames = list(rows[0].keys())
    ensure_dir(os.path.dirname(manifest_path) if os.path.dirname(manifest_path) else ".")

    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_bucket_summary(rows: List[Dict], title: str):
    counter = defaultdict(int)
    for row in rows:
        counter[(row["gt_label"], row["status"])] += 1

    print("=" * 80)
    print(title)
    print("=" * 80)
    for key in sorted(counter.keys()):
        print(f"{key}: {counter[key]}")
    print()


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="从 system1_scan_table.csv 自动挑选 train pool")
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry_run", action="store_true", help="只统计不复制")
    args = parser.parse_args()

    random.seed(args.seed)

    rows = load_scan_rows(args.input_csv)
    print(f"[*] 从 {args.input_csv} 读取有效样本数: {len(rows)}")
    print_bucket_summary(rows, "扫描总表 bucket 分布")

    grouped = group_by_bucket_and_subset(rows)

    selected_rows = []

    print("[*] 开始按目标 bucket 自动采样 ...")
    for bucket, target_n in DEFAULT_TARGET_BUCKETS.items():
        subset_to_rows = grouped.get(bucket, {})
        available_n = sum(len(v) for v in subset_to_rows.values())

        print(f"    - {bucket}: 目标={target_n}, 可用={available_n}, 子来源数={len(subset_to_rows)}")

        if available_n == 0:
            print(f"      [!] 跳过，当前 bucket 没有样本")
            continue

        chosen = round_robin_select(
            subset_to_rows=subset_to_rows,
            target_n=min(target_n, available_n),
            seed=args.seed + hash(bucket) % 10000
        )
        selected_rows.extend(chosen)

    print_bucket_summary(selected_rows, "已选中样本 bucket 分布")

    if args.dry_run:
        print("[*] dry_run 模式，不复制图片。")
        return

    ensure_dir(args.output_dir)
    manifest_rows = copy_selected_rows(selected_rows, args.output_dir)
    save_manifest(manifest_rows, args.manifest)

    print(f"[+] 已复制样本到: {args.output_dir}")
    print(f"[+] manifest 已保存到: {args.manifest}")

    # 额外打印 subset 分布
    subset_counter = defaultdict(int)
    for row in manifest_rows:
        subset_counter[(row["dataset_name"], row["subset_name"], row["gt_label"], row["status"])] += 1

    print("=" * 80)
    print("选中样本来源分布（dataset, subset, label, status）")
    print("=" * 80)
    for key in sorted(subset_counter.keys()):
        print(f"{key}: {subset_counter[key]}")


if __name__ == "__main__":
    main()