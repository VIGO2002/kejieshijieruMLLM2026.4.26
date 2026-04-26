"""
build_fusion_training_table_nos2.py

功能：
1. 加载 System 1 特征表
2. 调用训练好的 NoEvidence Expert，为所有样本生成 noevidence_fake_score
3. 生成 noevidence_fake_score_masked（仅在 NoEvidence 样本中激活）
4. 不合并任何 System 2 特征
5. 输出 fusion table（w/o System 2 消融版），供 Fusion Router 训练使用

用途：
- 做 Fusion w/o System 2 消融实验
- 验证当前性能是否主要来自：
  System 1 + NoEvidence Expert + Learned Fusion Router
"""

import os
import json
import argparse
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 默认配置
# ============================================================

DEFAULT_S1_CSV = "system1_feature_table_train_pool.csv"
DEFAULT_EXPERT_MODEL = "noevidence_expert_rf_core10.pkl"
DEFAULT_EXPERT_FEATURES = "noevidence_feature_list_rf_core10.json"
DEFAULT_OUTPUT_CSV = "fusion_training_table_nos2.csv"


# ============================================================
# 工具函数
# ============================================================

def ensure_parent_dir(path: str):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_gt_label(x: Any) -> str:
    s = str(x).strip().lower()
    if s == "fake":
        return "Fake"
    if s == "real":
        return "Real"
    return "Unknown"


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="构建 Fusion Table（w/o System 2 消融版）"
    )
    parser.add_argument("--s1_csv", type=str, default=DEFAULT_S1_CSV, help="System 1 特征表路径")
    parser.add_argument("--expert_model", type=str, default=DEFAULT_EXPERT_MODEL, help="NoEvidence Expert 模型路径")
    parser.add_argument("--expert_feats", type=str, default=DEFAULT_EXPERT_FEATURES, help="NoEvidence Expert 特征列表路径")
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV, help="输出融合表路径")
    args = parser.parse_args()

    # --------------------------------------------------------
    # 1. 加载 System 1 特征表
    # --------------------------------------------------------
    print(f"[*] 正在加载 System 1 特征表: {args.s1_csv}")
    if not os.path.exists(args.s1_csv):
        raise FileNotFoundError(f"[!] 找不到 S1 CSV: {args.s1_csv}")

    df_final = pd.read_csv(args.s1_csv, encoding="utf-8-sig")
    print(f"[*] S1 样本数: {len(df_final)}")

    if "case_name" not in df_final.columns:
        raise ValueError("[!] S1 CSV 缺少 case_name。")

    # --------------------------------------------------------
    # 2. 注入 NoEvidence Expert 分数
    # --------------------------------------------------------
    if os.path.exists(args.expert_model) and os.path.exists(args.expert_feats):
        print(f"[*] 正在调用 NoEvidence Expert: {args.expert_model}")
        expert = joblib.load(args.expert_model)
        expert_features = load_json(args.expert_feats)

        missing_feats = [f for f in expert_features if f not in df_final.columns]
        if missing_feats:
            raise ValueError(
                "[!] S1 CSV 缺少 Expert 所需特征列:\n" +
                "\n".join([f"    - {x}" for x in missing_feats])
            )

        X_expert = df_final[expert_features].fillna(0.0)

        if hasattr(expert, "predict_proba"):
            df_final["noevidence_fake_score"] = expert.predict_proba(X_expert)[:, 1]
        else:
            pred = expert.predict(X_expert)
            df_final["noevidence_fake_score"] = pred.astype(float)

        print("[+] NoEvidence Expert 分数已注入。")
    else:
        print("[!] 未找到 Expert 模型或特征列表，noevidence_fake_score 将置为 0。")
        df_final["noevidence_fake_score"] = 0.0

    # --------------------------------------------------------
    # 3. 构建 NoEvidence 掩码
    # --------------------------------------------------------
    if "status" in df_final.columns:
        df_final["is_noevidence_status"] = (df_final["status"].astype(str) == "NoEvidence").astype(int)
        df_final["is_high_status"] = (df_final["status"].astype(str) == "HighConfidenceEvidence").astype(int)
        df_final["is_weak_status"] = (df_final["status"].astype(str) == "WeakEvidence").astype(int)
        df_final["is_none_status"] = (df_final["status"].astype(str) == "NoEvidence").astype(int)
    else:
        # 兼容旧表结构
        df_final["is_noevidence_status"] = df_final["status_none"].fillna(0).astype(int) if "status_none" in df_final.columns else 0
        df_final["is_high_status"] = df_final["status_high"].fillna(0).astype(int) if "status_high" in df_final.columns else 0
        df_final["is_weak_status"] = df_final["status_weak"].fillna(0).astype(int) if "status_weak" in df_final.columns else 0
        df_final["is_none_status"] = df_final["status_none"].fillna(0).astype(int) if "status_none" in df_final.columns else 0

    # 仅在 NoEvidence 样本中激活 Expert 分数
    df_final["noevidence_fake_score_masked"] = (
        df_final["noevidence_fake_score"] * df_final["is_noevidence_status"]
    )

    # --------------------------------------------------------
    # 4. 标签清洗
    # --------------------------------------------------------
    if "gt_label" not in df_final.columns:
        raise ValueError("[!] S1 CSV 缺少 gt_label，无法构建训练标签。")

    df_final["gt_label_norm"] = df_final["gt_label"].apply(normalize_gt_label)
    df_final["target"] = df_final["gt_label_norm"].map({"Fake": 1, "Real": 0})

    # --------------------------------------------------------
    # 5. 统一填补缺失
    # --------------------------------------------------------
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df_final.select_dtypes(include=["object"]).columns.tolist()

    df_final[numeric_cols] = df_final[numeric_cols].fillna(0.0)
    df_final[object_cols] = df_final[object_cols].fillna("")

    # --------------------------------------------------------
    # 6. 保存
    # --------------------------------------------------------
    ensure_parent_dir(args.output_csv)
    df_final.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("\n[+] w/o System 2 消融版 Fusion Table 构建成功")
    print(f"[*] 输出路径: {args.output_csv}")
    print(f"[*] 样本总数: {len(df_final)}")
    print(f"[*] 总列数  : {len(df_final.columns)}")

    preview_cols = list(df_final.columns[:20])
    print(f"[*] 前 20 列示例: {preview_cols}")

    added_cols = [
        "noevidence_fake_score",
        "is_noevidence_status",
        "noevidence_fake_score_masked",
        "is_high_status",
        "is_weak_status",
        "is_none_status",
        "target",
    ]
    print(f"[*] 新增 / 关键列: {added_cols}")


if __name__ == "__main__":
    main()