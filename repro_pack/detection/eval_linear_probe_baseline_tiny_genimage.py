# eval_linear_probe_baseline_tiny_genimage.py

import os
import json
import argparse
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def save_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_gt_label(x: str) -> str:
    s = str(x).strip().lower()
    if s == "fake":
        return "Fake"
    if s == "real":
        return "Real"
    return "Unknown"


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def compute_binary_metrics(y_true, y_pred, y_prob_fake) -> Dict:
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob_fake))
    except Exception:
        metrics["roc_auc"] = None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def compute_group_stats(df: pd.DataFrame, group_col: str) -> Dict:
    out = {}
    if group_col not in df.columns:
        return out

    for g, sub in df.groupby(group_col):
        if len(sub) == 0:
            continue

        y_true = sub["target"].astype(int).tolist()
        y_pred = sub["baseline_pred"].astype(int).tolist()
        y_prob = sub["fake_prob"].astype(float).tolist()

        fake_count = int((sub["target"] == 1).sum())
        real_count = int((sub["target"] == 0).sum())

        stats = {
            "count": int(len(sub)),
            "fake_count": fake_count,
            "real_count": real_count,
        }
        stats.update(compute_binary_metrics(y_true, y_pred, y_prob))
        out[str(g)] = stats

    return out


def maybe_compare_with_fullchain(df_base: pd.DataFrame, fullchain_csv: str) -> Dict:
    if not fullchain_csv or (not os.path.exists(fullchain_csv)):
        return {}

    df_fc = pd.read_csv(fullchain_csv, encoding="utf-8-sig")

    required = {"case_name", "router_pred_label", "router_prob_fake"}
    if not required.issubset(set(df_fc.columns)):
        return {}

    merged = df_base.merge(
        df_fc[["case_name", "router_pred_label", "router_prob_fake"]],
        on="case_name",
        how="inner"
    )

    if len(merged) == 0:
        return {}

    merged["router_pred"] = merged["router_pred_label"].map({"Real": 0, "Fake": 1})

    y_true = merged["target"].astype(int).tolist()

    baseline_metrics = compute_binary_metrics(
        y_true,
        merged["baseline_pred"].astype(int).tolist(),
        merged["fake_prob"].astype(float).tolist()
    )
    fullchain_metrics = compute_binary_metrics(
        y_true,
        merged["router_pred"].astype(int).tolist(),
        merged["router_prob_fake"].astype(float).tolist()
    )

    delta = {}
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        bv = baseline_metrics.get(k, None)
        fv = fullchain_metrics.get(k, None)
        if bv is None or fv is None:
            delta[k] = None
        else:
            delta[k] = float(fv - bv)

    compare = {
        "num_compared_samples": int(len(merged)),
        "baseline_metrics": baseline_metrics,
        "fullchain_metrics": fullchain_metrics,
        "delta_fullchain_minus_baseline": delta,
    }
    return compare


def main():
    parser = argparse.ArgumentParser(description="Evaluate bare linear_probe baseline on Tiny-GenImage")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="System1 feature table CSV (must contain gt_label and fake_prob)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on fake_prob"
    )
    parser.add_argument(
        "--fullchain_predictions_csv",
        type=str,
        default="",
        help="Optional: predictions.csv from full detection chain for side-by-side comparison"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tiny_genimage_eval/results_linear_probe_baseline",
        help="Output directory"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[*] Loading input_csv: {args.input_csv}")
    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    print(f"[*] Rows: {len(df)}")

    required_cols = {"gt_label", "fake_prob"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[!] input_csv missing columns: {sorted(list(missing))}")

    if "case_name" not in df.columns:
        # 给一个兜底，避免 merge 时出问题
        df["case_name"] = [f"row_{i}" for i in range(len(df))]

    df["gt_label"] = df["gt_label"].apply(normalize_gt_label)
    df = df[df["gt_label"].isin(["Fake", "Real"])].copy()
    df["target"] = df["gt_label"].map({"Real": 0, "Fake": 1}).astype(int)

    if "status" not in df.columns:
        if "status_high" in df.columns and "status_weak" in df.columns and "status_none" in df.columns:
            def infer_status(row):
                if int(row.get("status_high", 0)) == 1:
                    return "HighConfidenceEvidence"
                if int(row.get("status_weak", 0)) == 1:
                    return "WeakEvidence"
                if int(row.get("status_none", 0)) == 1:
                    return "NoEvidence"
                return "Unknown"
            df["status"] = df.apply(infer_status, axis=1)
        else:
            df["status"] = "Unknown"

    df["fake_prob"] = df["fake_prob"].apply(safe_float)
    df["baseline_pred"] = (df["fake_prob"] >= args.threshold).astype(int)
    df["baseline_pred_label"] = df["baseline_pred"].map({0: "Real", 1: "Fake"})
    df["correct"] = (df["baseline_pred"] == df["target"]).astype(int)

    # Overall
    overall_metrics = compute_binary_metrics(
        df["target"].astype(int).tolist(),
        df["baseline_pred"].astype(int).tolist(),
        df["fake_prob"].astype(float).tolist()
    )
    overall_metrics["num_samples"] = int(len(df))
    overall_metrics["num_real"] = int((df["target"] == 0).sum())
    overall_metrics["num_fake"] = int((df["target"] == 1).sum())
    overall_metrics["threshold"] = float(args.threshold)

    # Group stats
    group_stats = {
        "by_status": compute_group_stats(df, "status"),
        "by_gt_label": compute_group_stats(df, "gt_label"),
    }

    noe_df = df[df["status"] == "NoEvidence"].copy()
    if len(noe_df) > 0:
        noe_metrics = compute_binary_metrics(
            noe_df["target"].astype(int).tolist(),
            noe_df["baseline_pred"].astype(int).tolist(),
            noe_df["fake_prob"].astype(float).tolist()
        )
        noe_metrics["count"] = int(len(noe_df))
        group_stats["NoEvidence_subset"] = noe_metrics
    else:
        group_stats["NoEvidence_subset"] = {}

    # Optional comparison with full chain
    compare_fullchain = maybe_compare_with_fullchain(df, args.fullchain_predictions_csv)

    # Save outputs
    pred_cols = []
    for c in [
        "case_name",
        "dataset",
        "split",
        "gt_label",
        "status",
        "fake_prob",
        "baseline_pred_label",
        "correct",
        "image_path",
    ]:
        if c in df.columns:
            pred_cols.append(c)

    pred_csv = os.path.join(args.output_dir, "predictions_baseline.csv")
    wrong_csv = os.path.join(args.output_dir, "wrong_cases_baseline.csv")
    metrics_json = os.path.join(args.output_dir, "metrics_baseline.json")
    group_json = os.path.join(args.output_dir, "group_stats_baseline.json")
    compare_json = os.path.join(args.output_dir, "compare_with_fullchain.json")

    df[pred_cols].to_csv(pred_csv, index=False, encoding="utf-8-sig")
    df[df["correct"] == 0][pred_cols].to_csv(wrong_csv, index=False, encoding="utf-8-sig")
    save_json(metrics_json, overall_metrics)
    save_json(group_json, group_stats)
    save_json(compare_json, compare_fullchain)

    print("[+] Linear-probe baseline evaluation finished")
    print(f"[*] predictions_baseline.csv : {pred_csv}")
    print(f"[*] wrong_cases_baseline.csv : {wrong_csv}")
    print(f"[*] metrics_baseline.json    : {metrics_json}")
    print(f"[*] group_stats_baseline.json: {group_json}")
    print(f"[*] compare_with_fullchain   : {compare_json}")
    print("[*] Overall baseline metrics:")
    print(json.dumps(overall_metrics, indent=2, ensure_ascii=False))

    if compare_fullchain:
        print("[*] Fullchain vs baseline delta:")
        print(json.dumps(compare_fullchain["delta_fullchain_minus_baseline"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()