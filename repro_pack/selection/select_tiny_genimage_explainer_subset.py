#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 Tiny-GenImage 的检测结果中，自动筛选 50+ 张解释评测子集。

输入:
1) full chain predictions.csv
2) linear probe baseline predictions.csv
3) system1 feature table csv
4) exported case 根目录 (每个 case 一个子目录)

输出:
- selected_manifest.csv
- summary.json
- cases/  (默认给每个选中的 case 建软链接)

推荐用途:
- 为后续 Phase A / Phase B / rule-based checker 构建 50+ 张解释评测集
"""

import os
import json
import math
import shutil
import argparse
from collections import Counter, defaultdict

import pandas as pd


# =========================================================
# 基础工具
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[!] 找不到 CSV: {path}")
    return pd.read_csv(path)


def norm_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"1", "fake", "ai", "generated", "synth", "synthetic"}:
        return "Fake"
    if s in {"0", "real", "natural", "authentic"}:
        return "Real"
    return str(x).strip()


def norm_status(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"highconfidenceevidence", "high", "high_evidence"}:
        return "HighConfidenceEvidence"
    if s in {"weakevidence", "weak", "weak_evidence"}:
        return "WeakEvidence"
    if s in {"noevidence", "none", "no_evidence"}:
        return "NoEvidence"
    return str(x).strip()


def norm_bool(x):
    if pd.isna(x):
        return None
    if isinstance(x, bool):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None


def maybe_float(x, default=None):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def pick_col(df: pd.DataFrame, candidates, required=True, name_hint="column"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"[!] 找不到列 {name_hint}，候选={candidates}，现有列={list(df.columns)}")
    return None


def add_softlink_or_copy(src: str, dst: str, mode="symlink"):
    if os.path.lexists(dst):
        return
    if mode == "none":
        return
    if mode == "copy":
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return

    # default: symlink
    try:
        os.symlink(src, dst)
    except Exception:
        # 若软链接失败，自动降级为 copy
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


# =========================================================
# 列标准化
# =========================================================

def prepare_fullchain_df(path: str) -> pd.DataFrame:
    df = safe_read_csv(path).copy()

    case_col = pick_col(df, ["case_name", "case", "id"], True, "fullchain.case_name")
    gt_col = pick_col(df, ["gt_label", "gt", "label", "y_true", "target"], True, "fullchain.gt_label")
    pred_col = pick_col(
        df,
        ["final_pred_label", "fusion_pred_label", "pred_label", "router_pred_label", "prediction"],
        True,
        "fullchain.pred_label"
    )
    correct_col = pick_col(
        df,
        ["final_correct", "correct", "is_correct", "cls_correct", "router_correct"],
        required=False,
        name_hint="fullchain.correct"
    )
    prob_col = pick_col(
        df,
        ["router_prob_fake", "pred_prob_fake", "prob_fake", "score_fake", "final_prob_fake"],
        required=False,
        name_hint="fullchain.prob_fake"
    )

    out = pd.DataFrame()
    out["case_name"] = df[case_col].astype(str)
    out["gt_label"] = df[gt_col].map(norm_label)
    out["full_pred_label"] = df[pred_col].map(norm_label)

    if correct_col is not None:
        out["full_correct"] = df[correct_col].map(norm_bool)
    else:
        out["full_correct"] = (out["gt_label"] == out["full_pred_label"])

    if prob_col is not None:
        out["full_prob_fake"] = df[prob_col].apply(maybe_float)
    else:
        out["full_prob_fake"] = None

    return out


def prepare_baseline_df(path: str) -> pd.DataFrame:
    df = safe_read_csv(path).copy()

    case_col = pick_col(df, ["case_name", "case", "id"], True, "baseline.case_name")
    gt_col = pick_col(df, ["gt_label", "gt", "label", "y_true", "target"], required=False, name_hint="baseline.gt_label")
    pred_col = pick_col(
        df,
        ["baseline_pred_label", "lp_pred_label", "pred_label", "linear_probe_pred_label", "prediction"],
        True,
        "baseline.pred_label"
    )
    correct_col = pick_col(
        df,
        ["baseline_correct", "lp_correct", "correct", "is_correct"],
        required=False,
        name_hint="baseline.correct"
    )
    prob_col = pick_col(
        df,
        ["baseline_prob_fake", "lp_prob_fake", "pred_prob_fake", "prob_fake", "score_fake"],
        required=False,
        name_hint="baseline.prob_fake"
    )

    out = pd.DataFrame()
    out["case_name"] = df[case_col].astype(str)
    if gt_col is not None:
        out["gt_label_baseline"] = df[gt_col].map(norm_label)
    else:
        out["gt_label_baseline"] = None

    out["baseline_pred_label"] = df[pred_col].map(norm_label)

    if correct_col is not None:
        out["baseline_correct"] = df[correct_col].map(norm_bool)
    else:
        # 如果没提供 correct，那后续 merge 后再算
        out["baseline_correct"] = None

    if prob_col is not None:
        out["baseline_prob_fake"] = df[prob_col].apply(maybe_float)
    else:
        out["baseline_prob_fake"] = None

    return out


def prepare_s1_df(path: str) -> pd.DataFrame:
    df = safe_read_csv(path).copy()

    case_col = pick_col(df, ["case_name", "case", "id"], True, "s1.case_name")
    gt_col = pick_col(df, ["gt_label", "gt", "label", "y_true", "target"], required=False, name_hint="s1.gt_label")
    status_col = pick_col(df, ["status", "system1_status"], True, "s1.status")
    dataset_col = pick_col(df, ["dataset"], required=False, name_hint="s1.dataset")
    source_col = pick_col(df, ["source_type", "source", "generator", "subset"], required=False, name_hint="s1.source_type")
    file_col = pick_col(df, ["filename", "file_name", "image_name"], required=False, name_hint="s1.filename")

    out = pd.DataFrame()
    out["case_name"] = df[case_col].astype(str)
    if gt_col is not None:
        out["gt_label_s1"] = df[gt_col].map(norm_label)
    else:
        out["gt_label_s1"] = None
    out["status"] = df[status_col].map(norm_status)

    out["dataset"] = df[dataset_col].astype(str) if dataset_col is not None else ""
    out["source_type"] = df[source_col].astype(str) if source_col is not None else ""
    out["filename"] = df[file_col].astype(str) if file_col is not None else ""

    return out


# =========================================================
# 多样性采样
# =========================================================

def build_interest_score(df: pd.DataFrame) -> pd.Series:
    """
    分数越高越优先。
    用于每个 bucket 内部排序。
    """
    score = pd.Series([0.0] * len(df), index=df.index)

    if "status" in df.columns:
        score += df["status"].map({
            "NoEvidence": 3.0,
            "WeakEvidence": 2.0,
            "HighConfidenceEvidence": 1.0
        }).fillna(0.0)

    if "full_correct" in df.columns:
        score += df["full_correct"].map(lambda x: 1.0 if x else 0.0)

    if "baseline_correct" in df.columns:
        # baseline 错但 full 对，很有价值
        mask = (df["baseline_correct"] == False) & (df["full_correct"] == True)
        score += mask.map(lambda x: 2.0 if x else 0.0)

    if "full_prob_fake" in df.columns:
        # 偏靠近 0.5 的更“有意思”（更容易体现解释与不确定性）
        score += df["full_prob_fake"].apply(
            lambda x: 0.0 if pd.isna(x) else (1.0 - min(abs(float(x) - 0.5) / 0.5, 1.0))
        )

    return score


def diverse_sample(df: pd.DataFrame, n: int, group_col: str = None) -> pd.DataFrame:
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()

    tmp = df.copy()
    tmp["_interest_score"] = build_interest_score(tmp)

    # 没有 group 列，直接按分数排序取前 n
    if group_col is None or group_col not in tmp.columns or tmp[group_col].fillna("").eq("").all():
        tmp = tmp.sort_values(by=["_interest_score", "case_name"], ascending=[False, True])
        return tmp.head(n).drop(columns=["_interest_score"])

    # 有 group 列，做 round-robin 多样性采样
    groups = {}
    for g, gdf in tmp.groupby(group_col):
        gg = gdf.sort_values(by=["_interest_score", "case_name"], ascending=[False, True]).copy()
        groups[g] = gg.reset_index(drop=True)

    selected_rows = []
    cursors = {g: 0 for g in groups.keys()}

    group_order = sorted(groups.keys(), key=lambda x: len(groups[x]), reverse=True)

    while len(selected_rows) < n:
        progressed = False
        for g in group_order:
            idx = cursors[g]
            if idx < len(groups[g]):
                selected_rows.append(groups[g].iloc[idx])
                cursors[g] += 1
                progressed = True
                if len(selected_rows) >= n:
                    break
        if not progressed:
            break

    if not selected_rows:
        return tmp.iloc[0:0].copy()

    out = pd.DataFrame(selected_rows).drop(columns=["_interest_score"], errors="ignore")
    # 去重（理论上不会重复，但保险）
    out = out.drop_duplicates(subset=["case_name"]).reset_index(drop=True)
    return out


# =========================================================
# 主逻辑
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True, help="完整检测主链 predictions.csv")
    parser.add_argument("--baseline_csv", required=True, help="linear probe baseline predictions.csv")
    parser.add_argument("--s1_csv", required=True, help="System1 feature table csv")
    parser.add_argument("--exports_dir", required=True, help="已导出的 case 根目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")

    # 默认目标数量（合计 60）
    parser.add_argument("--n_fake_high_correct", type=int, default=6)
    parser.add_argument("--n_fake_weak_correct", type=int, default=6)
    parser.add_argument("--n_fake_none_correct", type=int, default=10)
    parser.add_argument("--n_real_none_correct", type=int, default=10)
    parser.add_argument("--n_false_positive", type=int, default=8)
    parser.add_argument("--n_false_negative", type=int, default=8)
    parser.add_argument("--n_lp_wrong_full_right", type=int, default=6)
    parser.add_argument("--n_lp_right_full_wrong", type=int, default=6)

    parser.add_argument("--diversity_col", default="source_type", help="多样性采样分组列，默认 source_type")
    parser.add_argument("--link_mode", default="symlink", choices=["symlink", "copy", "none"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    cases_dir = os.path.join(args.output_dir, "cases")
    ensure_dir(cases_dir)

    print("[*] 加载 full chain predictions ...")
    full_df = prepare_fullchain_df(args.pred_csv)

    print("[*] 加载 baseline predictions ...")
    base_df = prepare_baseline_df(args.baseline_csv)

    print("[*] 加载 System 1 feature table ...")
    s1_df = prepare_s1_df(args.s1_csv)

    print("[*] 合并表格 ...")
    df = full_df.merge(s1_df, on="case_name", how="left")
    df = df.merge(base_df, on="case_name", how="left")

    # 统一 gt
    df["gt_label"] = df["gt_label"].fillna(df["gt_label_s1"]).fillna(df["gt_label_baseline"])
    df["gt_label"] = df["gt_label"].map(norm_label)

    # baseline_correct 若缺失，则用 gt/pred 自动算
    mask_missing_base_correct = df["baseline_correct"].isna()
    df.loc[mask_missing_base_correct, "baseline_correct"] = (
        df.loc[mask_missing_base_correct, "baseline_pred_label"].map(norm_label)
        == df.loc[mask_missing_base_correct, "gt_label"].map(norm_label)
    )

    # 标准化
    df["status"] = df["status"].map(norm_status)
    df["full_pred_label"] = df["full_pred_label"].map(norm_label)
    df["baseline_pred_label"] = df["baseline_pred_label"].map(norm_label)
    df["full_correct"] = df["full_correct"].map(lambda x: bool(x) if pd.notna(x) else False)
    df["baseline_correct"] = df["baseline_correct"].map(lambda x: bool(x) if pd.notna(x) else False)

    # 检查导出目录
    df["original_case_dir"] = df["case_name"].apply(lambda x: os.path.join(args.exports_dir, x))
    df["case_dir_exists"] = df["original_case_dir"].apply(os.path.isdir)
    missing_case_dirs = int((~df["case_dir_exists"]).sum())
    if missing_case_dirs > 0:
        print(f"[!] 警告: 有 {missing_case_dirs} 个 case 在 exports_dir 中找不到对应目录，将自动跳过。")

    df = df[df["case_dir_exists"]].copy().reset_index(drop=True)

    # 定义 bucket
    bucket_specs = [
        ("fake_none_correct",
         (df["gt_label"] == "Fake") &
         (df["status"] == "NoEvidence") &
         (df["full_correct"] == True),
         args.n_fake_none_correct),

        ("real_none_correct",
         (df["gt_label"] == "Real") &
         (df["status"] == "NoEvidence") &
         (df["full_correct"] == True),
         args.n_real_none_correct),

        ("false_positive",
         (df["gt_label"] == "Real") &
         (df["full_correct"] == False),
         args.n_false_positive),

        ("false_negative",
         (df["gt_label"] == "Fake") &
         (df["full_correct"] == False),
         args.n_false_negative),

        ("lp_wrong_full_right",
         (df["baseline_correct"] == False) &
         (df["full_correct"] == True),
         args.n_lp_wrong_full_right),

        ("lp_right_full_wrong",
         (df["baseline_correct"] == True) &
         (df["full_correct"] == False),
         args.n_lp_right_full_wrong),

        ("fake_high_correct",
         (df["gt_label"] == "Fake") &
         (df["status"] == "HighConfidenceEvidence") &
         (df["full_correct"] == True),
         args.n_fake_high_correct),

        ("fake_weak_correct",
         (df["gt_label"] == "Fake") &
         (df["status"] == "WeakEvidence") &
         (df["full_correct"] == True),
         args.n_fake_weak_correct),
    ]

    selected_chunks = []
    selected_case_names = set()

    pool_stats = {}
    selected_stats = {}
    shortage_stats = {}

    print("[*] 开始按 bucket 自动采样 ...")
    for bucket_name, cond, target_n in bucket_specs:
        pool = df[cond].copy()
        pool = pool[~pool["case_name"].isin(selected_case_names)].copy()

        pool_stats[bucket_name] = int(len(pool))
        print(f"    - {bucket_name}: 目标={target_n}, 可用={len(pool)}")

        sampled = diverse_sample(pool, target_n, group_col=args.diversity_col)
        sampled = sampled.copy()
        sampled["subset_bucket"] = bucket_name

        selected_chunks.append(sampled)
        selected_case_names.update(sampled["case_name"].tolist())

        selected_stats[bucket_name] = int(len(sampled))
        shortage_stats[bucket_name] = int(max(target_n - len(sampled), 0))

    if len(selected_chunks) == 0:
        raise RuntimeError("[!] 没有选中任何样本。")

    selected_df = pd.concat(selected_chunks, axis=0, ignore_index=True)
    selected_df = selected_df.drop_duplicates(subset=["case_name"]).reset_index(drop=True)

    # 生成软链接 / 拷贝
    print("[*] 生成 cases/ ...")
    for _, row in selected_df.iterrows():
        case_name = row["case_name"]
        src = row["original_case_dir"]
        dst = os.path.join(cases_dir, case_name)
        add_softlink_or_copy(src, dst, mode=args.link_mode)

    # 输出 manifest
    manifest_cols = [
        "case_name",
        "original_case_dir",
        "subset_bucket",
        "gt_label",
        "status",
        "full_pred_label",
        "full_correct",
        "full_prob_fake",
        "baseline_pred_label",
        "baseline_correct",
        "baseline_prob_fake",
        "dataset",
        "source_type",
        "filename",
    ]
    for c in manifest_cols:
        if c not in selected_df.columns:
            selected_df[c] = ""

    manifest_path = os.path.join(args.output_dir, "selected_manifest.csv")
    selected_df[manifest_cols].to_csv(manifest_path, index=False, encoding="utf-8-sig")

    # 汇总信息
    summary = {
        "input": {
            "pred_csv": args.pred_csv,
            "baseline_csv": args.baseline_csv,
            "s1_csv": args.s1_csv,
            "exports_dir": args.exports_dir,
        },
        "requested": {
            "fake_high_correct": args.n_fake_high_correct,
            "fake_weak_correct": args.n_fake_weak_correct,
            "fake_none_correct": args.n_fake_none_correct,
            "real_none_correct": args.n_real_none_correct,
            "false_positive": args.n_false_positive,
            "false_negative": args.n_false_negative,
            "lp_wrong_full_right": args.n_lp_wrong_full_right,
            "lp_right_full_wrong": args.n_lp_right_full_wrong,
        },
        "pool_stats": pool_stats,
        "selected_stats": selected_stats,
        "shortage_stats": shortage_stats,
        "selected_total": int(len(selected_df)),
        "status_dist": dict(Counter(selected_df["status"].astype(str).tolist())),
        "gt_dist": dict(Counter(selected_df["gt_label"].astype(str).tolist())),
        "bucket_dist": dict(Counter(selected_df["subset_bucket"].astype(str).tolist())),
        "source_type_dist": dict(Counter(selected_df["source_type"].astype(str).tolist())),
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[+] Tiny-GenImage 解释子集筛选完成")
    print(f"[*] manifest: {manifest_path}")
    print(f"[*] summary : {summary_path}")
    print(f"[*] cases dir: {cases_dir}")
    print(f"[*] 总计选中: {len(selected_df)}")


if __name__ == "__main__":
    main()