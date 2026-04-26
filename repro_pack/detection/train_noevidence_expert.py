"""
train_noevidence_expert.py

功能：
1. 加载 System 1 特征表（由 build_system1_feature_table.py 生成）
2. 过滤出 NoEvidence 样本
3. 用连续证据特征训练 NoEvidence Expert（无证据专家）
4. 支持参数化对照实验：
   - --model_type rf / xgb
   - --feature_mode core10 / full33
5. 若样本量允许，执行 Stratified K-Fold 交叉验证
6. 在全部 NoEvidence 数据上训练最终模型
7. 导出模型、特征列表、评估指标、特征重要性
"""

import os
import json
import argparse
import warnings
from typing import List, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 可选：优先 XGBoost，缺失则报错或回退
# ============================================================

HAS_XGB = True
try:
    import xgboost as xgb
except Exception:
    HAS_XGB = False
    xgb = None


# ============================================================
# 默认配置
# ============================================================

DEFAULT_INPUT_CSV = "system1_feature_table_train_pool.csv"
DEFAULT_OUTPUT_MODEL = "noevidence_expert.pkl"
DEFAULT_OUTPUT_FEATURES = "noevidence_feature_list.json"
DEFAULT_OUTPUT_METRICS = "noevidence_metrics.json"
DEFAULT_OUTPUT_IMPORTANCE = "noevidence_feature_importance.csv"

CORE_FEATURE_COLS = [
    "fake_prob",
    "best_score_max",
    "best_contrast_ratio",
    "best_token_contrast",
    "evidence_coverage",
    "num_raw_clusters",
    "num_singletons",
    "num_valid_clusters",
    "sys1_global_anomaly_prefix",
    "sys1_global_anomaly_gap",
]

OPTIONAL_FEATURE_COLS = [
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
]

MIN_FEATURES_REQUIRED = 6
DEFAULT_CV_FOLDS = 5
DEFAULT_RANDOM_SEED = 42


# ============================================================
# 工具函数
# ============================================================

def save_json(path: str, obj: Dict[str, Any]):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_gt_label(x: Any) -> str:
    s = str(x).strip().lower()
    if s == "fake":
        return "Fake"
    if s == "real":
        return "Real"
    return "Unknown"


def pick_feature_columns(df: pd.DataFrame, feature_mode: str) -> List[str]:
    core_cols = [c for c in CORE_FEATURE_COLS if c in df.columns]

    if feature_mode == "core10":
        return core_cols

    full_cols = core_cols[:]
    for c in OPTIONAL_FEATURE_COLS:
        if c in df.columns and c not in full_cols:
            full_cols.append(c)
    return full_cols


def build_model(
    y_train: np.ndarray,
    model_type: str,
    random_state: int = DEFAULT_RANDOM_SEED
):
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    scale_pos_weight = float(neg / max(pos, 1))

    if model_type == "xgb":
        if not HAS_XGB:
            raise RuntimeError("[!] 当前环境未安装 xgboost，请先执行: pip install xgboost")
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight
        )
        model_name = "XGBoost"
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
        model_name = "RandomForest"

    return model, model_name


def compute_feature_importance(model, feature_cols: List[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(feature_cols), dtype=float)

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return imp_df


def safe_cv_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    random_state: int = DEFAULT_RANDOM_SEED,
    max_folds: int = DEFAULT_CV_FOLDS
) -> Tuple[Dict[str, Any], bool]:
    y_np = y.to_numpy()
    class_counts = np.bincount(y_np)

    if len(class_counts) < 2:
        return {
            "cv_ran": False,
            "reason": "NoEvidence 子集只有一个类别，无法做交叉验证。"
        }, False

    min_class_count = int(class_counts.min())
    n_total = int(len(y_np))

    if n_total < 12 or min_class_count < 2:
        return {
            "cv_ran": False,
            "reason": f"样本过少（总数={n_total}, 最小类样本数={min_class_count}），跳过交叉验证。"
        }, False

    n_splits = min(max_folds, min_class_count)
    if n_splits < 2:
        return {
            "cv_ran": False,
            "reason": f"可用 folds 太少（n_splits={n_splits}），跳过交叉验证。"
        }, False

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model, model_name = build_model(y_np, model_type=model_type, random_state=random_state)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_res = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise"
    )

    metrics_dict = {
        "cv_ran": True,
        "cv_model_name": model_name,
        "cv_folds": int(n_splits),

        "cv_accuracy_mean": float(np.mean(cv_res["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_res["test_accuracy"])),

        "cv_precision_mean": float(np.mean(cv_res["test_precision"])),
        "cv_precision_std": float(np.std(cv_res["test_precision"])),

        "cv_recall_mean": float(np.mean(cv_res["test_recall"])),
        "cv_recall_std": float(np.std(cv_res["test_recall"])),

        "cv_f1_mean": float(np.mean(cv_res["test_f1"])),
        "cv_f1_std": float(np.std(cv_res["test_f1"])),

        "cv_roc_auc_mean": float(np.mean(cv_res["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_res["test_roc_auc"])),
    }
    return metrics_dict, True


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="训练 NoEvidence Expert（无证据专家）")
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV, help="System 1 特征表路径")
    parser.add_argument("--out_model", type=str, default=DEFAULT_OUTPUT_MODEL, help="模型输出路径")
    parser.add_argument("--out_features", type=str, default=DEFAULT_OUTPUT_FEATURES, help="特征列表输出路径")
    parser.add_argument("--out_metrics", type=str, default=DEFAULT_OUTPUT_METRICS, help="指标输出路径")
    parser.add_argument("--out_importance", type=str, default=DEFAULT_OUTPUT_IMPORTANCE, help="特征重要性输出路径")
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED)

    parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "xgb"], help="结构化模型类型")
    parser.add_argument("--feature_mode", type=str, default="full33", choices=["core10", "full33"], help="特征集合模式")
    args = parser.parse_args()

    print(f"[*] 正在加载特征表: {args.input_csv}")
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(
            f"[!] 找不到文件: {args.input_csv}\n"
            f"请先运行 build_system1_feature_table.py，或传入正确的 --input_csv"
        )

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    print(f"[*] 原始数据总行数: {len(df)}")

    # --------------------------------------------------------
    # 1. 标签清洗
    # --------------------------------------------------------
    if "gt_label" not in df.columns:
        raise ValueError("[!] 缺少 gt_label 列，无法训练。")

    df["gt_label_norm"] = df["gt_label"].apply(normalize_gt_label)
    df = df[df["gt_label_norm"].isin(["Fake", "Real"])].copy()

    if len(df) == 0:
        raise ValueError("[!] 清洗后没有任何 Fake/Real 样本。")

    # --------------------------------------------------------
    # 2. 过滤 NoEvidence
    # --------------------------------------------------------
    if "status" in df.columns:
        df_noe = df[df["status"].astype(str) == "NoEvidence"].copy()
    elif "status_none" in df.columns:
        df_noe = df[df["status_none"] == 1].copy()
    else:
        raise ValueError("[!] CSV 中缺少 status 或 status_none，无法筛 NoEvidence。")

    print(f"[*] 过滤后 NoEvidence 样本数: {len(df_noe)}")

    if len(df_noe) == 0:
        raise ValueError("[!] 没有 NoEvidence 样本，无法训练 NoEvidence Expert。")

    # --------------------------------------------------------
    # 3. 选择特征列
    # --------------------------------------------------------
    feature_cols = pick_feature_columns(df_noe, feature_mode=args.feature_mode)
    missing_core = [c for c in CORE_FEATURE_COLS if c not in df_noe.columns]

    if missing_core:
        print("[!] 警告：以下核心特征缺失，将自动跳过：")
        for c in missing_core:
            print(f"    - {c}")

    if len(feature_cols) < MIN_FEATURES_REQUIRED:
        raise ValueError(
            f"[!] 可用特征太少，仅有 {len(feature_cols)} 个。\n"
            f"当前可用特征: {feature_cols}"
        )

    print(f"[*] model_type   : {args.model_type}")
    print(f"[*] feature_mode : {args.feature_mode}")
    print(f"[*] 实际使用特征数: {len(feature_cols)}")
    for c in feature_cols:
        print(f"    - {c}")

    # 缺失值填 0
    df_noe[feature_cols] = df_noe[feature_cols].fillna(0.0)

    # 标签编码
    df_noe["target"] = df_noe["gt_label_norm"].map({"Real": 0, "Fake": 1})
    X = df_noe[feature_cols].copy()
    y = df_noe["target"].astype(int).copy()

    class_counts = y.value_counts().to_dict()
    print(f"[*] NoEvidence 子集类别分布: {class_counts}")

    if len(class_counts) < 2:
        raise ValueError("[!] NoEvidence 子集里只有一个类别，无法训练分类器。")

    # --------------------------------------------------------
    # 4. 交叉验证
    # --------------------------------------------------------
    print("[*] 尝试进行交叉验证评估 ...")
    cv_metrics, did_cv = safe_cv_evaluate(
        X=X,
        y=y,
        model_type=args.model_type,
        random_state=args.random_seed,
        max_folds=DEFAULT_CV_FOLDS
    )

    if did_cv:
        print("-" * 60)
        print("[+] Cross-Validation 结果")
        print(f"    Accuracy : {cv_metrics['cv_accuracy_mean']:.4f} ± {cv_metrics['cv_accuracy_std']:.4f}")
        print(f"    Precision: {cv_metrics['cv_precision_mean']:.4f} ± {cv_metrics['cv_precision_std']:.4f}")
        print(f"    Recall   : {cv_metrics['cv_recall_mean']:.4f} ± {cv_metrics['cv_recall_std']:.4f}")
        print(f"    F1       : {cv_metrics['cv_f1_mean']:.4f} ± {cv_metrics['cv_f1_std']:.4f}")
        print(f"    ROC AUC  : {cv_metrics['cv_roc_auc_mean']:.4f} ± {cv_metrics['cv_roc_auc_std']:.4f}")
        print("-" * 60)
    else:
        print(f"[!] 跳过交叉验证: {cv_metrics['reason']}")

    # --------------------------------------------------------
    # 5. 在全部 NoEvidence 数据上训练最终模型
    # --------------------------------------------------------
    model, model_name = build_model(
        y.to_numpy(),
        model_type=args.model_type,
        random_state=args.random_seed
    )
    print(f"[*] 正在训练最终模型: {model_name}")
    model.fit(X, y)

    y_pred_train = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_prob_train = model.predict_proba(X)[:, 1]
    else:
        y_prob_train = y_pred_train.astype(float)

    train_metrics = {
        "train_accuracy": float(accuracy_score(y, y_pred_train)),
        "train_precision": float(precision_score(y, y_pred_train, zero_division=0)),
        "train_recall": float(recall_score(y, y_pred_train, zero_division=0)),
        "train_f1": float(f1_score(y, y_pred_train, zero_division=0)),
        "train_roc_auc": float(roc_auc_score(y, y_prob_train)) if len(np.unique(y)) == 2 else None,
    }

    print("-" * 60)
    print("[+] Final Model 在全部 NoEvidence 数据上的训练内参考指标")
    print(f"    Train Accuracy : {train_metrics['train_accuracy']:.4f}")
    print(f"    Train Precision: {train_metrics['train_precision']:.4f}")
    print(f"    Train Recall   : {train_metrics['train_recall']:.4f}")
    print(f"    Train F1       : {train_metrics['train_f1']:.4f}")
    if train_metrics["train_roc_auc"] is not None:
        print(f"    Train ROC AUC  : {train_metrics['train_roc_auc']:.4f}")
    print("-" * 60)

    # --------------------------------------------------------
    # 6. 特征重要性
    # --------------------------------------------------------
    imp_df = compute_feature_importance(model, feature_cols)
    print("[*] 特征重要性 Top 15")
    print(imp_df.head(15).to_string(index=False))

    # --------------------------------------------------------
    # 7. 保存
    # --------------------------------------------------------
    out_dir = os.path.dirname(args.out_model)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    joblib.dump(model, args.out_model)

    with open(args.out_features, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    imp_df.to_csv(args.out_importance, index=False, encoding="utf-8-sig")

    metrics_dict = {
        "input_csv": args.input_csv,
        "model_name": model_name,
        "requested_model_type": args.model_type,
        "feature_mode": args.feature_mode,
        "has_xgboost": HAS_XGB,
        "n_total_noevidence": int(len(df_noe)),
        "n_fake": int(np.sum(y == 1)),
        "n_real": int(np.sum(y == 0)),
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "cv_metrics": cv_metrics,
        "final_train_metrics": train_metrics,
        "note": (
            "cv_metrics 是更值得参考的泛化指标；"
            "final_train_metrics 只是全部 NoEvidence 数据上的训练内 sanity check。"
        )
    }
    save_json(args.out_metrics, metrics_dict)

    print(f"[+] 模型已保存: {args.out_model}")
    print(f"[+] 特征列表已保存: {args.out_features}")
    print(f"[+] 特征重要性已保存: {args.out_importance}")
    print(f"[+] 指标已保存: {args.out_metrics}")
    print("[+] NoEvidence Expert 训练完成。")


if __name__ == "__main__":
    main()