"""
eval_noevidence_expert_on_holdout.py

功能：
1. 读取 holdout 的 System 1 特征表 CSV
2. 只筛选 NoEvidence 样本
3. 加载已经训练好的 noevidence_expert_train_pool.pkl
4. 按 feature list 对 holdout 样本打分
5. 输出：
   - 总体指标 JSON
   - 每样本预测 CSV
   - 错误样本 CSV
   - 按 dataset / subset 的分组统计 JSON
"""

import os
import csv
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


DEFAULT_HOLDOUT_CSV = "system1_feature_table_holdout_pool.csv"
DEFAULT_MODEL = "noevidence_expert_train_pool.pkl"
DEFAULT_FEATURE_LIST = "noevidence_feature_list_train_pool.json"

DEFAULT_OUT_JSON = "noevidence_eval_holdout.json"
DEFAULT_OUT_PRED_CSV = "noevidence_eval_holdout_predictions.csv"
DEFAULT_OUT_WRONG_CSV = "noevidence_eval_holdout_wrong.csv"
DEFAULT_OUT_GROUP_JSON = "noevidence_eval_holdout_group_stats.json"

VALID_LABELS = {"Fake", "Real"}


def ensure_parent_dir(path: str):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Dict[str, Any]):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[!] 找不到 CSV: {path}")

    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(x, default=0.0):
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def normalize_gt_label(x: Any) -> str:
    s = str(x).strip().lower()
    if s == "fake":
        return "Fake"
    if s == "real":
        return "Real"
    return "Unknown"


def is_noevidence_row(row: Dict[str, str]) -> bool:
    status = str(row.get("status", "")).strip()
    if status == "NoEvidence":
        return True
    status_none = str(row.get("status_none", "")).strip()
    if status_none in {"1", "1.0"}:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="使用现有 NoEvidence expert 在 holdout CSV 上打分评估")
    parser.add_argument("--holdout_csv", type=str, default=DEFAULT_HOLDOUT_CSV)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--feature_list", type=str, default=DEFAULT_FEATURE_LIST)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_pred_csv", type=str, default=DEFAULT_OUT_PRED_CSV)
    parser.add_argument("--out_wrong_csv", type=str, default=DEFAULT_OUT_WRONG_CSV)
    parser.add_argument("--out_group_json", type=str, default=DEFAULT_OUT_GROUP_JSON)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"[!] 找不到模型: {args.model}")
    if not os.path.exists(args.feature_list):
        raise FileNotFoundError(f"[!] 找不到特征列表: {args.feature_list}")

    feature_cols = load_json(args.feature_list)
    model = joblib.load(args.model)

    rows = load_csv_rows(args.holdout_csv)
    print(f"[*] 原始 holdout CSV 行数: {len(rows)}")

    filtered_rows = []
    for row in rows:
        gt = normalize_gt_label(row.get("gt_label", "Unknown"))
        if gt not in VALID_LABELS:
            continue
        if not is_noevidence_row(row):
            continue

        row["gt_label_norm"] = gt
        filtered_rows.append(row)

    print(f"[*] holdout 中 NoEvidence 样本数: {len(filtered_rows)}")
    if len(filtered_rows) == 0:
        raise RuntimeError("[!] holdout CSV 中没有 NoEvidence 样本。")

    # 构造 X / y
    X_list = []
    y_list = []
    pred_rows = []

    missing_feature_counter = defaultdict(int)

    for row in filtered_rows:
        feat_vec = []
        for feat in feature_cols:
            if feat not in row:
                missing_feature_counter[feat] += 1
            feat_vec.append(safe_float(row.get(feat, 0.0), 0.0))

        X_list.append(feat_vec)
        y_list.append(1 if row["gt_label_norm"] == "Fake" else 0)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    # 打分
    if hasattr(model, "predict_proba"):
        prob_fake = model.predict_proba(X)[:, 1]
    else:
        pred_raw = model.predict(X)
        prob_fake = pred_raw.astype(np.float32)

    pred_binary = (prob_fake >= args.threshold).astype(np.int64)

    # 指标
    acc = accuracy_score(y, pred_binary)
    prec = precision_score(y, pred_binary, zero_division=0)
    rec = recall_score(y, pred_binary, zero_division=0)
    f1 = f1_score(y, pred_binary, zero_division=0)
    auc = roc_auc_score(y, prob_fake) if len(np.unique(y)) == 2 else None

    tn, fp, fn, tp = confusion_matrix(y, pred_binary, labels=[0, 1]).ravel()

    # 每样本预测表
    wrong_rows = []
    for idx, row in enumerate(filtered_rows):
        pred_label = "Fake" if pred_binary[idx] == 1 else "Real"
        correct = int(pred_label == row["gt_label_norm"])

        out_row = {
            "case_name": row.get("case_name", ""),
            "filename": row.get("filename", ""),
            "dataset": row.get("dataset", row.get("dataset_name", "")),
            "source_type": row.get("source_type", ""),
            "subset_name": row.get("subset_name", ""),
            "gt_label": row["gt_label_norm"],
            "status": row.get("status", ""),
            "pred_label": pred_label,
            "pred_prob_fake": float(prob_fake[idx]),
            "threshold": float(args.threshold),
            "correct": correct,
        }
        pred_rows.append(out_row)

        if correct == 0:
            wrong_rows.append(out_row)

    pred_fieldnames = [
        "case_name",
        "filename",
        "dataset",
        "source_type",
        "subset_name",
        "gt_label",
        "status",
        "pred_label",
        "pred_prob_fake",
        "threshold",
        "correct",
    ]

    save_csv(args.out_pred_csv, pred_rows, pred_fieldnames)
    save_csv(args.out_wrong_csv, wrong_rows, pred_fieldnames)

    # 分组统计：按 dataset / subset
    group_stats = defaultdict(lambda: {"count": 0, "correct": 0, "fake_count": 0, "real_count": 0})
    for row in pred_rows:
        key = f"{row['dataset']}||{row['subset_name']}"
        group_stats[key]["count"] += 1
        group_stats[key]["correct"] += row["correct"]
        if row["gt_label"] == "Fake":
            group_stats[key]["fake_count"] += 1
        else:
            group_stats[key]["real_count"] += 1

    group_stats_out = []
    for key, val in sorted(group_stats.items()):
        dataset, subset_name = key.split("||", 1)
        group_stats_out.append({
            "dataset": dataset,
            "subset_name": subset_name,
            "count": val["count"],
            "correct": val["correct"],
            "accuracy": float(val["correct"] / max(val["count"], 1)),
            "fake_count": val["fake_count"],
            "real_count": val["real_count"],
        })

    save_json(args.out_group_json, {"groups": group_stats_out})

    summary = {
        "holdout_csv": args.holdout_csv,
        "model": args.model,
        "feature_list": args.feature_list,
        "threshold": args.threshold,
        "n_total_noevidence_holdout": int(len(filtered_rows)),
        "n_fake": int(np.sum(y == 1)),
        "n_real": int(np.sum(y == 0)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc) if auc is not None else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "missing_feature_counter": dict(missing_feature_counter),
        "prediction_csv": args.out_pred_csv,
        "wrong_csv": args.out_wrong_csv,
        "group_stats_json": args.out_group_json,
    }
    save_json(args.out_json, summary)

    print("=" * 90)
    print("[+] NoEvidence expert 在 holdout NoEvidence 子集上的结果")
    print("=" * 90)
    print(f"样本数      : {len(filtered_rows)}")
    print(f"Fake / Real : {int(np.sum(y == 1))} / {int(np.sum(y == 0))}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"F1          : {f1:.4f}")
    if auc is not None:
        print(f"ROC AUC     : {auc:.4f}")
    print(f"Confusion   : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("-" * 90)
    print(f"[+] summary json 已保存到: {args.out_json}")
    print(f"[+] prediction csv 已保存到: {args.out_pred_csv}")
    print(f"[+] wrong csv 已保存到: {args.out_wrong_csv}")
    print(f"[+] group json 已保存到: {args.out_group_json}")


if __name__ == "__main__":
    main()