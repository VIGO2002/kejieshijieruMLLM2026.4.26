import os
import csv
import json
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def norm_label(x):
    s = str(x).strip().lower()
    if s == "fake":
        return "Fake"
    if s == "real":
        return "Real"
    return x


def round_robin_select(rows, n, key_fields, seed=42):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k, "") for k in key_fields)
        groups[key].append(r)

    for k in groups:
        rng.shuffle(groups[k])

    ordered_keys = list(groups.keys())
    rng.shuffle(ordered_keys)

    out = []
    while len(out) < n:
        progressed = False
        for k in ordered_keys:
            if groups[k]:
                out.append(groups[k].pop())
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    return out


def safe_symlink(src, dst):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:
        # 某些环境不允许软链接，则退化为目录复制
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="自动筛选 12 张协议调试集")
    parser.add_argument("--pred_csv", type=str, default="fusion_router_rf_core10_holdout_predictions.csv")
    parser.add_argument("--wrong_csv", type=str, default="fusion_router_rf_core10_holdout_wrong.csv")
    parser.add_argument("--fusion_table", type=str, default="fusion_training_table_rf_core10_holdout.csv")
    parser.add_argument("--exports_dir", type=str, default="system2_holdout_pool_exports")
    parser.add_argument("--output_dir", type=str, default="protocol_debug_set")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_high_fake", type=int, default=2)
    parser.add_argument("--n_weak_fake", type=int, default=2)
    parser.add_argument("--n_noe_fake", type=int, default=2)
    parser.add_argument("--n_noe_real", type=int, default=2)
    parser.add_argument("--n_fp", type=int, default=2)
    parser.add_argument("--n_fn", type=int, default=2)
    args = parser.parse_args()

    preds = read_csv(args.pred_csv)
    wrongs = read_csv(args.wrong_csv)
    fusion_rows = read_csv(args.fusion_table)

    fusion_by_case = {r["case_name"]: r for r in fusion_rows}

    merged_preds = []
    for r in preds:
        case_name = r["case_name"]
        if case_name not in fusion_by_case:
            continue
        x = dict(r)
        x.update(fusion_by_case[case_name])
        x["gt_label"] = norm_label(x.get("gt_label", ""))
        x["pred_label"] = norm_label(x.get("pred_label", ""))
        merged_preds.append(x)

    merged_wrongs = []
    for r in wrongs:
        case_name = r["case_name"]
        if case_name not in fusion_by_case:
            continue
        x = dict(r)
        x.update(fusion_by_case[case_name])
        x["gt_label"] = norm_label(x.get("gt_label", ""))
        x["pred_label"] = norm_label(x.get("pred_label", ""))
        merged_wrongs.append(x)

    selected = []
    selected_case_names = set()

    def add_bucket(name, candidates, n):
        nonlocal selected, selected_case_names
        candidates = [c for c in candidates if c["case_name"] not in selected_case_names]
        chosen = round_robin_select(
            candidates,
            n,
            key_fields=["dataset", "source_type", "status"],
            seed=args.seed + len(selected)
        )
        for c in chosen:
            c["protocol_bucket"] = name
            selected.append(c)
            selected_case_names.add(c["case_name"])
        return len(chosen)

    # 正确样本：High Fake / Weak Fake / NoEvidence Fake / NoEvidence Real
    high_fake = [
        r for r in merged_preds
        if str(r.get("correct", "0")) in ["1", "1.0"]
        and r["gt_label"] == "Fake"
        and r.get("status", "") == "HighConfidenceEvidence"
    ]
    weak_fake = [
        r for r in merged_preds
        if str(r.get("correct", "0")) in ["1", "1.0"]
        and r["gt_label"] == "Fake"
        and r.get("status", "") == "WeakEvidence"
    ]
    noe_fake = [
        r for r in merged_preds
        if str(r.get("correct", "0")) in ["1", "1.0"]
        and r["gt_label"] == "Fake"
        and r.get("status", "") == "NoEvidence"
    ]
    noe_real = [
        r for r in merged_preds
        if str(r.get("correct", "0")) in ["1", "1.0"]
        and r["gt_label"] == "Real"
        and r.get("status", "") == "NoEvidence"
    ]

    # 错误样本：FP / FN
    fps = [
        r for r in merged_wrongs
        if r["gt_label"] == "Real" and r["pred_label"] == "Fake"
    ]
    fns = [
        r for r in merged_wrongs
        if r["gt_label"] == "Fake" and r["pred_label"] == "Real"
    ]

    print("[*] 候选池统计：")
    print(f"    High Fake      : {len(high_fake)}")
    print(f"    Weak Fake      : {len(weak_fake)}")
    print(f"    NoEvidence Fake: {len(noe_fake)}")
    print(f"    NoEvidence Real: {len(noe_real)}")
    print(f"    False Positive : {len(fps)}")
    print(f"    False Negative : {len(fns)}")

    add_bucket("High_Fake", high_fake, args.n_high_fake)
    add_bucket("Weak_Fake", weak_fake, args.n_weak_fake)
    add_bucket("NoEvidence_Fake", noe_fake, args.n_noe_fake)
    add_bucket("NoEvidence_Real", noe_real, args.n_noe_real)
    add_bucket("False_Positive", fps, args.n_fp)
    add_bucket("False_Negative", fns, args.n_fn)

    # 生成 case_dir，并创建链接
    out_cases_dir = os.path.join(args.output_dir, "cases")
    os.makedirs(out_cases_dir, exist_ok=True)

    manifest_rows = []
    missing_dirs = []
    for r in selected:
        case_name = r["case_name"]
        case_dir = os.path.join(args.exports_dir, case_name)
        linked_dir = os.path.join(out_cases_dir, case_name)

        if not os.path.isdir(case_dir):
            missing_dirs.append(case_name)
            continue

        safe_symlink(case_dir, linked_dir)

        manifest_rows.append({
            "protocol_bucket": r["protocol_bucket"],
            "case_name": case_name,
            "gt_label": r["gt_label"],
            "pred_label": r["pred_label"],
            "correct": r.get("correct", ""),
            "status": r.get("status", ""),
            "dataset": r.get("dataset", ""),
            "source_type": r.get("source_type", ""),
            "pred_prob_fake": r.get("pred_prob_fake", ""),
            "linked_case_dir": linked_dir,
            "original_case_dir": case_dir
        })

    manifest_path = os.path.join(args.output_dir, "selected_manifest.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")

    if manifest_rows:
        write_csv(
            manifest_path,
            manifest_rows,
            fieldnames=list(manifest_rows[0].keys())
        )

    summary = {
        "requested": {
            "High_Fake": args.n_high_fake,
            "Weak_Fake": args.n_weak_fake,
            "NoEvidence_Fake": args.n_noe_fake,
            "NoEvidence_Real": args.n_noe_real,
            "False_Positive": args.n_fp,
            "False_Negative": args.n_fn
        },
        "selected_count": len(manifest_rows),
        "selected_by_bucket": {
            k: sum(1 for r in manifest_rows if r["protocol_bucket"] == k)
            for k in [
                "High_Fake", "Weak_Fake", "NoEvidence_Fake",
                "NoEvidence_Real", "False_Positive", "False_Negative"
            ]
        },
        "missing_case_dirs": missing_dirs
    }
    write_json(summary_path, summary)

    print("\n[+] 协议调试集筛选完成")
    print(f"[*] manifest: {manifest_path}")
    print(f"[*] summary : {summary_path}")
    print(f"[*] case link dir: {out_cases_dir}")
    if missing_dirs:
        print(f"[!] 有 {len(missing_dirs)} 个 case 在 exports_dir 中未找到对应目录。")


if __name__ == "__main__":
    main()