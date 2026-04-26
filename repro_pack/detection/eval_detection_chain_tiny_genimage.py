# eval_detection_chain_tiny_genimage.py

import os
import json
import argparse
from typing import Dict, List

import joblib
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


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def normalize_gt_label(x: str) -> str:
    s = str(x).strip().lower()
    if s == "fake":
        return "Fake"
    if s == "real":
        return "Real"
    return "Unknown"


def ensure_required_columns(df: pd.DataFrame, cols: List[str], fill_value=0.0) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
    return df


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
        y_pred = sub["router_pred"].astype(int).tolist()
        y_prob = sub["router_prob_fake"].astype(float).tolist()

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection chain on Tiny-GenImage")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="System1 feature table CSV for Tiny-GenImage"
    )
    parser.add_argument(
        "--expert_model",
        type=str,
        default="noevidence_expert_rf_core10.pkl",
        help="Path to NoEvidence expert model"
    )
    parser.add_argument(
        "--expert_feats",
        type=str,
        default="noevidence_feature_list_rf_core10.json",
        help="Path to NoEvidence expert feature list JSON"
    )
    parser.add_argument(
        "--router_model",
        type=str,
        default="fusion_router_rf_core10_nos2.pkl",
        help="Path to fusion router model"
    )
    parser.add_argument(
        "--router_feats",
        type=str,
        default="fusion_router_rf_core10_nos2_feature_list.json",
        help="Path to fusion router feature list JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tiny_genimage_eval/results_detection_chain",
        help="Output directory"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[*] Loading input_csv: {args.input_csv}")
    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    print(f"[*] Rows: {len(df)}")

    if "gt_label" not in df.columns:
        raise ValueError("[!] input_csv must contain gt_label column")

    df["gt_label"] = df["gt_label"].apply(normalize_gt_label)
    df = df[df["gt_label"].isin(["Fake", "Real"])].copy()
    df["target"] = df["gt_label"].map({"Real": 0, "Fake": 1}).astype(int)

    # --------------------------------------------------
    # 1) Load NoEvidence expert and inject scores
    # --------------------------------------------------
    print(f"[*] Loading NoEvidence expert: {args.expert_model}")
    expert = joblib.load(args.expert_model)
    expert_feats = load_json(args.expert_feats)

    df = ensure_required_columns(df, expert_feats, fill_value=0.0)
    X_expert = df[expert_feats].fillna(0.0)

    if hasattr(expert, "predict_proba"):
        df["noevidence_fake_score"] = expert.predict_proba(X_expert)[:, 1]
    else:
        df["noevidence_fake_score"] = expert.predict(X_expert).astype(float)

    # 状态列兼容
    if "status" not in df.columns:
        if "status_none" in df.columns:
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

    df["is_noevidence_status"] = (df["status"] == "NoEvidence").astype(int)
    df["noevidence_fake_score_masked"] = (
        df["noevidence_fake_score"] * df["is_noevidence_status"]
    )

    # --------------------------------------------------
    # 2) Load fusion router
    # --------------------------------------------------
    print(f"[*] Loading fusion router: {args.router_model}")
    router = joblib.load(args.router_model)
    router_feats = load_json(args.router_feats)

    df = ensure_required_columns(df, router_feats, fill_value=0.0)
    X_router = df[router_feats].fillna(0.0)

    if hasattr(router, "predict_proba"):
        prob_fake = router.predict_proba(X_router)[:, 1]
    else:
        # 若没有 predict_proba，则退化为 predict
        pred_tmp = router.predict(X_router).astype(int)
        prob_fake = pred_tmp.astype(float)

    pred_fake = (prob_fake >= 0.5).astype(int)

    df["router_prob_fake"] = prob_fake
    df["router_pred"] = pred_fake
    df["router_pred_label"] = df["router_pred"].map({0: "Real", 1: "Fake"})
    df["correct"] = (df["router_pred"] == df["target"]).astype(int)

    # --------------------------------------------------
    # 3) Overall metrics
    # --------------------------------------------------
    y_true = df["target"].astype(int).tolist()
    y_pred = df["router_pred"].astype(int).tolist()
    y_prob = df["router_prob_fake"].astype(float).tolist()

    overall_metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    overall_metrics["num_samples"] = int(len(df))
    overall_metrics["num_real"] = int((df["target"] == 0).sum())
    overall_metrics["num_fake"] = int((df["target"] == 1).sum())

    # --------------------------------------------------
    # 4) Bucket-level stats
    # --------------------------------------------------
    group_stats = {
        "by_status": compute_group_stats(df, "status"),
        "by_gt_label": compute_group_stats(df, "gt_label"),
    }

    # 专门统计 NoEvidence 子集
    noe_df = df[df["status"] == "NoEvidence"].copy()
    if len(noe_df) > 0:
        noe_metrics = compute_binary_metrics(
            noe_df["target"].astype(int).tolist(),
            noe_df["router_pred"].astype(int).tolist(),
            noe_df["router_prob_fake"].astype(float).tolist(),
        )
        noe_metrics["count"] = int(len(noe_df))
        group_stats["NoEvidence_subset"] = noe_metrics
    else:
        group_stats["NoEvidence_subset"] = {}

    # --------------------------------------------------
    # 5) Save outputs
    # --------------------------------------------------
    pred_cols = []
    for c in [
        "case_name",
        "dataset",
        "split",
        "gt_label",
        "status",
        "fake_prob",
        "best_score_max",
        "best_token_contrast",
        "evidence_coverage",
        "sys1_global_anomaly_prefix",
        "sys1_global_anomaly_gap",
        "noevidence_fake_score",
        "noevidence_fake_score_masked",
        "router_prob_fake",
        "router_pred_label",
        "correct",
        "image_path",
    ]:
        if c in df.columns:
            pred_cols.append(c)

    pred_csv = os.path.join(args.output_dir, "predictions.csv")
    wrong_csv = os.path.join(args.output_dir, "wrong_cases.csv")
    metrics_json = os.path.join(args.output_dir, "metrics.json")
    group_json = os.path.join(args.output_dir, "group_stats.json")

    df[pred_cols].to_csv(pred_csv, index=False, encoding="utf-8-sig")
    df[df["correct"] == 0][pred_cols].to_csv(wrong_csv, index=False, encoding="utf-8-sig")
    save_json(metrics_json, overall_metrics)
    save_json(group_json, group_stats)

    print("[+] Tiny-GenImage detection-chain evaluation finished")
    print(f"[*] predictions.csv : {pred_csv}")
    print(f"[*] wrong_cases.csv : {wrong_csv}")
    print(f"[*] metrics.json    : {metrics_json}")
    print(f"[*] group_stats.json: {group_json}")
    print("[*] Overall metrics:")
    print(json.dumps(overall_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()