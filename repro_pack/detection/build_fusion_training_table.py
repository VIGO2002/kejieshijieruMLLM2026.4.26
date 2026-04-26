"""
build_fusion_training_table.py

功能：
1. 加载 System 1 特征表
2. 调用训练好的 NoEvidence Expert，为所有样本生成 noevidence_fake_score
3. 合并一个或多个 System 2 结果表（支持 ensemble）
4. 数值化类别特征（status / local_verdict / local_confidence）
5. 构建最终 fusion_training_table.csv，供 Fusion Router 训练使用

重要说明：
- 这个脚本默认面向 train_pool，而不是 holdout
- 不应把 holdout 的 System 2 结果拿来训练 Fusion Router
"""

import os
import json
import argparse
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 默认配置（请按 train_pool 使用）
# ============================================================

DEFAULT_S1_CSV = "system1_feature_table_train_pool.csv"
DEFAULT_EXPERT_MODEL = "noevidence_expert_train_pool.pkl"
DEFAULT_EXPERT_FEATURES = "noevidence_feature_list_train_pool.json"
DEFAULT_S2_CSVS = ["system2_train_pool_fusion_outputs_qwen_dualcall/fusion_summary_train_pool.csv"]
DEFAULT_OUTPUT_CSV = "fusion_training_table_train_pool.csv"


# ============================================================
# 编码函数
# ============================================================

def normalize_gt_label(x: Any) -> str:
    s = str(x).strip().lower()
    if s == "fake":
        return "Fake"
    if s == "real":
        return "Real"
    return "Unknown"


def encode_local_verdict(verdict_str) -> float:
    """
    当前体系优先兼容：
    supported / weak / none
    同时兼容旧字符串
    """
    if pd.isna(verdict_str):
        return 0.0
    v = str(verdict_str).lower().strip()

    if "supported" in v:
        return 1.0
    if "weak" in v:
        return 0.5
    if "none" in v:
        return 0.0
    if "refuted" in v:
        return -1.0
    if "inconclusive" in v:
        return 0.0
    return 0.0


def encode_local_conf(conf_str) -> float:
    if pd.isna(conf_str):
        return 0.0
    v = str(conf_str).lower().strip()
    if v == "high":
        return 1.0
    if v == "medium":
        return 0.5
    if v == "low":
        return 0.0
    return 0.0


def ensure_parent_dir(path: str):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_existing_col(df: pd.DataFrame, candidates: List[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="构建 Fusion Router 训练表")
    parser.add_argument("--s1_csv", type=str, default=DEFAULT_S1_CSV, help="System 1 特征表")
    parser.add_argument("--expert_model", type=str, default=DEFAULT_EXPERT_MODEL, help="NoEvidence 专家模型")
    parser.add_argument("--expert_feats", type=str, default=DEFAULT_EXPERT_FEATURES, help="NoEvidence 专家特征列表")
    parser.add_argument("--s2_csvs", nargs="+", default=DEFAULT_S2_CSVS, help="一个或多个 System 2 结果 CSV")
    parser.add_argument("--s2_names", nargs="*", default=None, help="可选：给多个 System 2 指定名字前缀，如 flash plus")
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
        raise ValueError("[!] S1 CSV 缺少 case_name，无法和 System 2 结果对齐。")

    # --------------------------------------------------------
    # 2. 注入 NoEvidence Expert 分数
    # --------------------------------------------------------
    if os.path.exists(args.expert_model) and os.path.exists(args.expert_feats):
        print(f"[*] 正在调用 NoEvidence 专家模型: {args.expert_model}")
        expert = joblib.load(args.expert_model)
        expert_features = load_json(args.expert_feats)

        missing_expert_feats = [f for f in expert_features if f not in df_final.columns]
        if missing_expert_feats:
            raise ValueError(
                "[!] S1 CSV 缺少专家模型需要的特征列:\n" +
                "\n".join([f"    - {x}" for x in missing_expert_feats])
            )

        X_expert = df_final[expert_features].fillna(0.0)

        if hasattr(expert, "predict_proba"):
            df_final["noevidence_fake_score"] = expert.predict_proba(X_expert)[:, 1]
        else:
            pred = expert.predict(X_expert)
            df_final["noevidence_fake_score"] = pred.astype(float)

        print("[+] NoEvidence 专家得分已注入。")
    else:
        print("[!] 未找到专家模型或特征列表，noevidence_fake_score 将置 0。")
        df_final["noevidence_fake_score"] = 0.0

    # 只在 NoEvidence 样本上激活的 masked 版本
    if "status" in df_final.columns:
        df_final["is_noevidence_status"] = (df_final["status"].astype(str) == "NoEvidence").astype(int)
    elif "status_none" in df_final.columns:
        df_final["is_noevidence_status"] = df_final["status_none"].fillna(0).astype(int)
    else:
        df_final["is_noevidence_status"] = 0

    df_final["noevidence_fake_score_masked"] = (
        df_final["noevidence_fake_score"] * df_final["is_noevidence_status"]
    )

    # --------------------------------------------------------
    # 3. 合并一个或多个 System 2 结果表
    # --------------------------------------------------------
    if args.s2_names is not None and len(args.s2_names) > 0:
        if len(args.s2_names) != len(args.s2_csvs):
            raise ValueError("[!] --s2_names 的数量必须和 --s2_csvs 一致。")

    for i, s2_path in enumerate(args.s2_csvs):
        print(f"[*] 正在合并 System 2 结果 ({i+1}/{len(args.s2_csvs)}): {s2_path}")
        if not os.path.exists(s2_path):
            print(f"[!] 跳过不存在的文件: {s2_path}")
            continue

        df_s2 = pd.read_csv(s2_path, encoding="utf-8-sig")

        if "case_name" not in df_s2.columns:
            raise ValueError(f"[!] S2 CSV 缺少 case_name: {s2_path}")

        if args.s2_names is not None and len(args.s2_names) > 0:
            prefix = f"s2_{args.s2_names[i]}_"
        else:
            prefix = f"s2_{i}_"

        # 自动兼容不同 runner 版本字段名
        gvis_col = pick_existing_col(df_s2, [
            "sys2_global_visual_suspicion_score",
            "gvis",
            "global_visual_suspicion_score"
        ])
        local_verdict_col = pick_existing_col(df_s2, [
            "sys2_local_evidence_verdict",
            "local_evidence_verdict"
        ])
        local_conf_col = pick_existing_col(df_s2, [
            "sys2_local_evidence_confidence",
            "local_evidence_confidence"
        ])

        keep_cols = ["case_name"]
        rename_map = {}

        if gvis_col is not None:
            keep_cols.append(gvis_col)
            rename_map[gvis_col] = f"{prefix}gvis"

        if local_verdict_col is not None:
            keep_cols.append(local_verdict_col)
            rename_map[local_verdict_col] = f"{prefix}local_verdict"

        if local_conf_col is not None:
            keep_cols.append(local_conf_col)
            rename_map[local_conf_col] = f"{prefix}local_conf"

        df_s2_sub = df_s2[keep_cols].copy()
        df_s2_sub.rename(columns=rename_map, inplace=True)

        verdict_name = f"{prefix}local_verdict"
        conf_name = f"{prefix}local_conf"
        gvis_name = f"{prefix}gvis"

        if verdict_name in df_s2_sub.columns:
            df_s2_sub[f"{verdict_name}_val"] = df_s2_sub[verdict_name].apply(encode_local_verdict)

        if conf_name in df_s2_sub.columns:
            df_s2_sub[f"{conf_name}_val"] = df_s2_sub[conf_name].apply(encode_local_conf)

        # merge
        df_final = pd.merge(df_final, df_s2_sub, on="case_name", how="left")

        # gvis 缺失标记 + 缺省填充
        if gvis_name in df_final.columns:
            df_final[f"{gvis_name}_missing"] = df_final[gvis_name].isna().astype(int)
            df_final[gvis_name] = df_final[gvis_name].fillna(0.0)

        # local verdict/conf 缺失标记
        if verdict_name in df_final.columns:
            df_final[f"{verdict_name}_missing"] = df_final[verdict_name].isna().astype(int)
            if f"{verdict_name}_val" in df_final.columns:
                df_final[f"{verdict_name}_val"] = df_final[f"{verdict_name}_val"].fillna(0.0)

        if conf_name in df_final.columns:
            df_final[f"{conf_name}_missing"] = df_final[conf_name].isna().astype(int)
            if f"{conf_name}_val" in df_final.columns:
                df_final[f"{conf_name}_val"] = df_final[f"{conf_name}_val"].fillna(0.0)

    # --------------------------------------------------------
    # 4. 状态 one-hot / 数值化
    # --------------------------------------------------------
    if "status" in df_final.columns:
        df_final["is_high_status"] = (df_final["status"] == "HighConfidenceEvidence").astype(int)
        df_final["is_weak_status"] = (df_final["status"] == "WeakEvidence").astype(int)
        df_final["is_none_status"] = (df_final["status"] == "NoEvidence").astype(int)
    else:
        if "status_high" in df_final.columns:
            df_final["is_high_status"] = df_final["status_high"].fillna(0).astype(int)
        if "status_weak" in df_final.columns:
            df_final["is_weak_status"] = df_final["status_weak"].fillna(0).astype(int)
        if "status_none" in df_final.columns:
            df_final["is_none_status"] = df_final["status_none"].fillna(0).astype(int)

    # --------------------------------------------------------
    # 5. 标签清洗
    # --------------------------------------------------------
    if "gt_label" in df_final.columns:
        df_final["gt_label_norm"] = df_final["gt_label"].apply(normalize_gt_label)
        df_final["target"] = df_final["gt_label_norm"].map({"Fake": 1, "Real": 0})
    else:
        raise ValueError("[!] S1 CSV 缺少 gt_label，无法构建训练标签 target。")

    # --------------------------------------------------------
    # 6. 统一填补其余缺失
    # --------------------------------------------------------
    # 对 object / numeric 特征剩余缺失填 0
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
    df_final[numeric_cols] = df_final[numeric_cols].fillna(0.0)

    # 保留文本元信息，缺失补空串
    object_cols = df_final.select_dtypes(include=["object"]).columns.tolist()
    df_final[object_cols] = df_final[object_cols].fillna("")

    # --------------------------------------------------------
    # 7. 保存
    # --------------------------------------------------------
    ensure_parent_dir(args.output_csv)
    df_final.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("\n[+] Fusion 训练表构建成功")
    print(f"[*] 输出路径: {args.output_csv}")
    print(f"[*] 样本总数: {len(df_final)}")
    print(f"[*] 总列数  : {len(df_final.columns)}")

    preview_cols = list(df_final.columns[:20])
    print(f"[*] 前 20 列示例: {preview_cols}")

    # 额外提示有哪些 System 2 特征被并入
    s2_feature_cols = [c for c in df_final.columns if c.startswith("s2_")]
    print(f"[*] 已并入的 System 2 特征列数: {len(s2_feature_cols)}")
    if len(s2_feature_cols) > 0:
        print(f"[*] System 2 特征示例: {s2_feature_cols[:15]}")


if __name__ == "__main__":
    main()