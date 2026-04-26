"""
train_fusion_router.py

功能：
1. 加载 fusion train table（融合训练表）
2. 自动发现数值特征，避免把字符串列误送入模型
3. 训练 Learned Fusion Router（学习式融合路由器）
4. 在训练集上做 Stratified K-Fold（分层 K 折）交叉验证
5. 可选：在独立 holdout fusion table（留出测试融合表）上评估
6. 导出模型、特征列表、指标、特征重要性、holdout 预测结果

推荐：
- 第一轮固定 router 底座为 RandomForest（随机森林）
- 先比较两条主线：
    A. RF + core10 expert
    B. RF + full33 expert
"""

import os
import json
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 可选：支持 XGBoost（极端梯度提升）
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

DEFAULT_TRAIN_CSV = "fusion_training_table_train.csv"
DEFAULT_HOLDOUT_CSV = "fusion_training_table_holdout.csv"

DEFAULT_OUT_MODEL = "fusion_router.pkl"
DEFAULT_OUT_FEATURES = "fusion_router_feature_list.json"
DEFAULT_OUT_METRICS = "fusion_router_metrics.json"
DEFAULT_OUT_IMPORTANCE = "fusion_router_feature_importance.csv"
DEFAULT_OUT_HOLDOUT_PRED = "fusion_router_holdout_predictions.csv"
DEFAULT_OUT_HOLDOUT_WRONG = "fusion_router_holdout_wrong.csv"

DEFAULT_RANDOM_SEED = 42
DEFAULT_CV_FOLDS = 5

# 不作为训练特征的列
EXCLUDE_EXACT = {
    "case_name",
    "case_tag",
    "dataset",
    "source_type",
    "filename",
    "gt_label",
    "gt_label_norm",
    "pred_label",
    "status",
    "target",
    "cls_correct",
}

# 包含这些模式的列也剔除
EXCLUDE_CONTAINS = [
    "img_path",
    "copied_path",
]

# 原始文本列不要直接进模型
# 但 *_val 和 *_missing 这种编码列要保留
def should_exclude_col(col: str) -> bool:
    if col in EXCLUDE_EXACT:
        return True

    for pat in EXCLUDE_CONTAINS:
        if pat in col:
            return True

    if col.endswith("_rationale"):
        return True

    if "local_verdict" in col and (not col.endswith("_val")) and (not col.endswith("_missing")):
        return True

    if "local_conf" in col and (not col.endswith("_val")) and (not col.endswith("_missing")):
        return True

    return False


# ============================================================
# 工具函数
# ============================================================

def ensure_parent_dir(path: str):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def save_json(path: str, obj):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_gt_label(x):
    s = str(x).strip().lower()
    if s == "fake":
        return "Fake"
    if s == "real":
        return "Real"
    return "Unknown"


def discover_numeric_feature_cols(df: pd.DataFrame):
    feature_cols = []
    for col in df.columns:
        if should_exclude_col(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def build_model(y_train: np.ndarray, model_type: str, random_state: int = DEFAULT_RANDOM_SEED):
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    scale_pos_weight = float(neg / max(pos, 1))

    if model_type == "xgb":
        if not HAS_XGB:
            raise RuntimeError("[!] 当前环境未安装 xgboost，请先执行: pip install xgboost")
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
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
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
        model_name = "RandomForest"

    return model, model_name


def safe_cv_evaluate(X, y, model_type: str, random_state: int, max_folds: int):
    y_np = y.to_numpy()
    class_counts = np.bincount(y_np)

    if len(class_counts) < 2:
        return {"cv_ran": False, "reason": "训练集只有一个类别，无法做交叉验证。"}, False

    min_class_count = int(class_counts.min())
    n_total = int(len(y_np))

    if n_total < 20 or min_class_count < 2:
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
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
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

    metrics = {
        "cv_ran": True,
        "cv_model_name": model_name,
        "cv_folds": int(n_splits),
    }

    for metric in scoring.keys():
        metrics[f"cv_{metric}_mean"] = float(np.mean(cv_res[f"test_{metric}"]))
        metrics[f"cv_{metric}_std"] = float(np.std(cv_res[f"test_{metric}"]))

    return metrics, True


def compute_feature_importance(model, feature_cols):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(feature_cols), dtype=float)

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return imp_df


def align_holdout_features(df_holdout: pd.DataFrame, feature_cols):
    aligned = pd.DataFrame(index=df_holdout.index)
    for col in feature_cols:
        if col in df_holdout.columns:
            aligned[col] = df_holdout[col]
        else:
            aligned[col] = 0.0
    return aligned.fillna(0.0)


def evaluate_on_holdout(model, df_holdout: pd.DataFrame, feature_cols, threshold: float = 0.5):
    if "gt_label" not in df_holdout.columns:
        raise ValueError("[!] holdout CSV 缺少 gt_label，无法评估。")

    df_eval = df_holdout.copy()
    df_eval["gt_label_norm"] = df_eval["gt_label"].apply(normalize_gt_label)
    df_eval = df_eval[df_eval["gt_label_norm"].isin(["Fake", "Real"])].copy()

    if len(df_eval) == 0:
        raise ValueError("[!] holdout 中没有可用的 Fake/Real 样本。")

    X_holdout = align_holdout_features(df_eval, feature_cols)
    y_true = df_eval["gt_label_norm"].map({"Real": 0, "Fake": 1}).astype(int).to_numpy()

    if hasattr(model, "predict_proba"):
        prob_fake = model.predict_proba(X_holdout)[:, 1]
    else:
        pred_tmp = model.predict(X_holdout)
        prob_fake = pred_tmp.astype(float)

    y_pred = (prob_fake >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, prob_fake) if len(np.unique(y_true)) == 2 else None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pred_rows = pd.DataFrame({
        "case_name": df_eval["case_name"] if "case_name" in df_eval.columns else "",
        "filename": df_eval["filename"] if "filename" in df_eval.columns else "",
        "dataset": df_eval["dataset"] if "dataset" in df_eval.columns else "",
        "source_type": df_eval["source_type"] if "source_type" in df_eval.columns else "",
        "gt_label": df_eval["gt_label_norm"],
        "pred_prob_fake": prob_fake,
        "pred_label": np.where(y_pred == 1, "Fake", "Real"),
        "correct": (y_pred == y_true).astype(int),
        "threshold": threshold,
    })

    wrong_rows = pred_rows[pred_rows["correct"] == 0].copy()

    metrics = {
        "n_holdout_samples": int(len(df_eval)),
        "n_holdout_fake": int(np.sum(y_true == 1)),
        "n_holdout_real": int(np.sum(y_true == 0)),
        "holdout_accuracy": float(acc),
        "holdout_precision": float(prec),
        "holdout_recall": float(rec),
        "holdout_f1": float(f1),
        "holdout_roc_auc": float(auc) if auc is not None else None,
        "holdout_confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    }
    return metrics, pred_rows, wrong_rows


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="训练 Learned Fusion Router（学习式融合路由器）")
    parser.add_argument("--train_csv", type=str, required=True, help="train fusion table 路径")
    parser.add_argument("--holdout_csv", type=str, default=None, help="可选：独立 holdout fusion table 路径")
    parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "xgb"], help="router 底座模型")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    parser.add_argument("--out_model", type=str, default=DEFAULT_OUT_MODEL)
    parser.add_argument("--out_features", type=str, default=DEFAULT_OUT_FEATURES)
    parser.add_argument("--out_metrics", type=str, default=DEFAULT_OUT_METRICS)
    parser.add_argument("--out_importance", type=str, default=DEFAULT_OUT_IMPORTANCE)
    parser.add_argument("--out_holdout_pred", type=str, default=DEFAULT_OUT_HOLDOUT_PRED)
    parser.add_argument("--out_holdout_wrong", type=str, default=DEFAULT_OUT_HOLDOUT_WRONG)

    args = parser.parse_args()

    # 1. 加载 train csv
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"[!] 找不到训练表: {args.train_csv}")

    print(f"[*] 正在加载训练 fusion 表: {args.train_csv}")
    df_train = pd.read_csv(args.train_csv, encoding="utf-8-sig")

    if "gt_label" not in df_train.columns:
        raise ValueError("[!] train_csv 缺少 gt_label。")

    df_train["gt_label_norm"] = df_train["gt_label"].apply(normalize_gt_label)
    df_train = df_train[df_train["gt_label_norm"].isin(["Fake", "Real"])].copy()
    df_train["target"] = df_train["gt_label_norm"].map({"Fake": 1, "Real": 0})

    if len(df_train) < 20:
        raise ValueError(f"[!] 训练样本量过小 ({len(df_train)})，不建议训练 fusion router。")

    print(f"[*] 训练样本数: {len(df_train)}")
    print(f"[*] 训练类别分布: {df_train['target'].value_counts().to_dict()}")

    # 2. 自动识别数值特征
    feature_cols = discover_numeric_feature_cols(df_train)
    if "target" in feature_cols:
        feature_cols.remove("target")

    if len(feature_cols) == 0:
        raise ValueError("[!] 没有识别到可用数值特征。")

    print(f"[*] 成功识别 {len(feature_cols)} 个数值特征。")
    print(f"[*] 前 12 个特征示例: {feature_cols[:12]}")

    X_train = df_train[feature_cols].fillna(0.0)
    y_train = df_train["target"].astype(int)

    # 3. 交叉验证
    print("[*] 开始训练集上的 Stratified K-Fold 交叉验证 ...")
    cv_metrics, did_cv = safe_cv_evaluate(
        X=X_train,
        y=y_train,
        model_type=args.model_type,
        random_state=args.seed,
        max_folds=DEFAULT_CV_FOLDS
    )

    if did_cv:
        print("=" * 60)
        print(f"[+] Fusion Router ({args.model_type}) 交叉验证结果")
        print("-" * 60)
        print(f"Accuracy : {cv_metrics['cv_accuracy_mean']:.4f} ± {cv_metrics['cv_accuracy_std']:.4f}")
        print(f"ROC AUC  : {cv_metrics['cv_roc_auc_mean']:.4f} ± {cv_metrics['cv_roc_auc_std']:.4f}")
        print(f"F1       : {cv_metrics['cv_f1_mean']:.4f} ± {cv_metrics['cv_f1_std']:.4f}")
        print(f"Precision: {cv_metrics['cv_precision_mean']:.4f} ± {cv_metrics['cv_precision_std']:.4f}")
        print(f"Recall   : {cv_metrics['cv_recall_mean']:.4f} ± {cv_metrics['cv_recall_std']:.4f}")
        print("=" * 60)
    else:
        print(f"[!] 跳过交叉验证: {cv_metrics['reason']}")

    # 4. 全量训练
    model, model_name = build_model(
        y_train.to_numpy(),
        model_type=args.model_type,
        random_state=args.seed
    )
    print(f"[*] 正在全量训练 Fusion Router: {model_name}")
    model.fit(X_train, y_train)

    # 5. 训练集内 sanity check
    train_pred = model.predict(X_train)
    if hasattr(model, "predict_proba"):
        train_prob = model.predict_proba(X_train)[:, 1]
    else:
        train_prob = train_pred.astype(float)

    train_metrics = {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "train_precision": float(precision_score(y_train, train_pred, zero_division=0)),
        "train_recall": float(recall_score(y_train, train_pred, zero_division=0)),
        "train_f1": float(f1_score(y_train, train_pred, zero_division=0)),
        "train_roc_auc": float(roc_auc_score(y_train, train_prob)) if len(np.unique(y_train)) == 2 else None,
    }

    print("-" * 60)
    print("[+] 训练集内参考指标")
    print(f"Train Accuracy : {train_metrics['train_accuracy']:.4f}")
    print(f"Train Precision: {train_metrics['train_precision']:.4f}")
    print(f"Train Recall   : {train_metrics['train_recall']:.4f}")
    print(f"Train F1       : {train_metrics['train_f1']:.4f}")
    if train_metrics["train_roc_auc"] is not None:
        print(f"Train ROC AUC  : {train_metrics['train_roc_auc']:.4f}")
    print("-" * 60)

    # 6. 特征重要性
    imp_df = compute_feature_importance(model, feature_cols)
    print("[*] Fusion Router Top 15 特征")
    print(imp_df.head(15).to_string(index=False))

    # 7. holdout 评估（可选）
    holdout_metrics = {}
    if args.holdout_csv is not None:
        if not os.path.exists(args.holdout_csv):
            raise FileNotFoundError(f"[!] 找不到 holdout 表: {args.holdout_csv}")

        print(f"[*] 正在加载 holdout fusion 表: {args.holdout_csv}")
        df_holdout = pd.read_csv(args.holdout_csv, encoding="utf-8-sig")

        holdout_metrics, pred_rows, wrong_rows = evaluate_on_holdout(
            model=model,
            df_holdout=df_holdout,
            feature_cols=feature_cols,
            threshold=0.5
        )

        print("=" * 60)
        print("[+] Fusion Router 在独立 holdout 上的结果")
        print("-" * 60)
        print(f"Accuracy : {holdout_metrics['holdout_accuracy']:.4f}")
        print(f"Precision: {holdout_metrics['holdout_precision']:.4f}")
        print(f"Recall   : {holdout_metrics['holdout_recall']:.4f}")
        print(f"F1       : {holdout_metrics['holdout_f1']:.4f}")
        if holdout_metrics["holdout_roc_auc"] is not None:
            print(f"ROC AUC  : {holdout_metrics['holdout_roc_auc']:.4f}")
        cm = holdout_metrics["holdout_confusion_matrix"]
        print(f"Confusion: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
        print("=" * 60)

        ensure_parent_dir(args.out_holdout_pred)
        pred_rows.to_csv(args.out_holdout_pred, index=False, encoding="utf-8-sig")
        wrong_rows.to_csv(args.out_holdout_wrong, index=False, encoding="utf-8-sig")

    # 8. 保存
    ensure_parent_dir(args.out_model)
    joblib.dump(model, args.out_model)

    with open(args.out_features, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    imp_df.to_csv(args.out_importance, index=False, encoding="utf-8-sig")

    metrics = {
        "model_type": model_name,
        "requested_model_type": args.model_type,
        "train_csv": args.train_csv,
        "holdout_csv": args.holdout_csv,
        "n_train_samples": int(len(df_train)),
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "cv_metrics": cv_metrics,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "top_features": imp_df.head(20).to_dict(orient="records"),
    }

    save_json(args.out_metrics, metrics)

    print(f"[+] Fusion Router 模型已保存: {args.out_model}")
    print(f"[+] 特征列表已保存: {args.out_features}")
    print(f"[+] 特征重要性已保存: {args.out_importance}")
    print(f"[+] 指标 JSON 已保存: {args.out_metrics}")
    if args.holdout_csv is not None:
        print(f"[+] holdout 预测明细已保存: {args.out_holdout_pred}")
        print(f"[+] holdout 错误样本已保存: {args.out_holdout_wrong}")


if __name__ == "__main__":
    main()