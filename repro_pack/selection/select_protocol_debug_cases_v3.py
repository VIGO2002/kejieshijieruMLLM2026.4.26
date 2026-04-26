
import os
import csv
import json
import random
import shutil
import argparse
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
    return str(x).strip()


def norm_correct(x):
    s = str(x).strip()
    return s in ["1", "1.0", "True", "true"]


def safe_symlink(src, dst):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def diverse_select(rows, n, seed=42):
    """
    尽量按 dataset/source_type/status 做多样化轮转采样
    """
    rng = random.Random(seed)
    grouped = defaultdict(list)

    for r in rows:
        key = (
            r.get("dataset", ""),
            r.get("source_type", ""),
            r.get("status", ""),
        )
        grouped[key].append(r)

    keys = list(grouped.keys())
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(grouped[k])

    out = []
    while len(out) < n:
        progressed = False
        for k in keys:
            if grouped[k]:
                out.append(grouped[k].pop())
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    return out


def dedupe_by_case(rows):
    seen = set()
    out = []
    for r in rows:
        c = r.get("case_name", "")
        if c and c not in seen:
            out.append(r)
            seen.add(c)
    return out


def merge_with_fusion(rows, fusion_by_case):
    merged = []
    for r in rows:
        case_name = r.get("case_name", "")
        if case_name not in fusion_by_case:
            continue
        x = dict(r)
        x.update(fusion_by_case[case_name])
        x["gt_label"] = norm_label(x.get("gt_label", ""))
        x["pred_label"] = norm_label(x.get("pred_label", ""))
        x["correct_bool"] = norm_correct(x.get("correct", ""))
        merged.append(x)
    return dedupe_by_case(merged)


def build_fp_fn_from_preds(merged_preds):
    fps = []
    fns = []
    for r in merged_preds:
        gt = r.get("gt_label", "")
        pred = r.get("pred_label", "")
        correct = r.get("correct_bool", False)

        if (not correct) and gt == "Real" and pred == "Fake":
            fps.append(r)
        if (not correct) and gt == "Fake" and pred == "Real":
            fns.append(r)

    return dedupe_by_case(fps), dedupe_by_case(fns)


def attach_case_dirs(rows, exports_dir):
    ok_rows = []
    missing = []

    for r in rows:
        case_name = r["case_name"]
        case_dir = os.path.join(exports_dir, case_name)
        if os.path.isdir(case_dir):
            rr = dict(r)
            rr["original_case_dir"] = case_dir
            ok_rows.append(rr)
        else:
            missing.append(case_name)

    return ok_rows, missing


def main():
    parser = argparse.ArgumentParser(description="补齐 2 张 False Positive，并生成 16–18 张完整版协议调试集")
    parser.add_argument("--pred_csv", type=str, default="fusion_router_rf_core10_holdout_predictions.csv")
    parser.add_argument("--wrong_csv", type=str, default="fusion_router_rf_core10_holdout_wrong.csv")
    parser.add_argument("--fusion_table", type=str, default="fusion_training_table_rf_core10_holdout.csv")
    parser.add_argument("--exports_dir", type=str, default="system2_holdout_pool_exports")
    parser.add_argument("--output_dir", type=str, default="protocol_debug_set_v3")
    parser.add_argument("--seed", type=int, default=42)

    # 目标桶
    parser.add_argument("--n_high_fake", type=int, default=3)
    parser.add_argument("--n_weak_fake", type=int, default=3)
    parser.add_argument("--n_noe_fake", type=int, default=3)
    parser.add_argument("--n_noe_real", type=int, default=3)
    parser.add_argument("--n_fp", type=int, default=2)
    parser.add_argument("--n_fn", type=int, default=2)

    # 总量控制
    parser.add_argument("--min_total", type=int, default=16)
    parser.add_argument("--max_total", type=int, default=18)

    args = parser.parse_args()

    preds = read_csv(args.pred_csv)
    wrongs = read_csv(args.wrong_csv)
    fusion_rows = read_csv(args.fusion_table)
    fusion_by_case = {r["case_name"]: r for r in fusion_rows}

    merged_preds = merge_with_fusion(preds, fusion_by_case)
    merged_wrongs = merge_with_fusion(wrongs, fusion_by_case)

    # 重新从 pred_csv 推断 FP/FN，比只信 wrong_csv 更稳
    fp_from_preds, fn_from_preds = build_fp_fn_from_preds(merged_preds)

    # wrong_csv 作为补充池
    fp_from_wrongs = [r for r in merged_wrongs if r["gt_label"] == "Real" and r["pred_label"] == "Fake"]
    fn_from_wrongs = [r for r in merged_wrongs if r["gt_label"] == "Fake" and r["pred_label"] == "Real"]

    fps = dedupe_by_case(fp_from_preds + fp_from_wrongs)
    fns = dedupe_by_case(fn_from_preds + fn_from_wrongs)

    # 正确样本桶
    high_fake = [
        r for r in merged_preds
        if r["correct_bool"]
        and r["gt_label"] == "Fake"
        and r.get("status", "") == "HighConfidenceEvidence"
    ]
    weak_fake = [
        r for r in merged_preds
        if r["correct_bool"]
        and r["gt_label"] == "Fake"
        and r.get("status", "") == "WeakEvidence"
    ]
    noe_fake = [
        r for r in merged_preds
        if r["correct_bool"]
        and r["gt_label"] == "Fake"
        and r.get("status", "") == "NoEvidence"
    ]
    noe_real = [
        r for r in merged_preds
        if r["correct_bool"]
        and r["gt_label"] == "Real"
        and r.get("status", "") == "NoEvidence"
    ]

    # 先过滤出有目录的样本
    high_fake, miss_h = attach_case_dirs(high_fake, args.exports_dir)
    weak_fake, miss_w = attach_case_dirs(weak_fake, args.exports_dir)
    noe_fake, miss_nf = attach_case_dirs(noe_fake, args.exports_dir)
    noe_real, miss_nr = attach_case_dirs(noe_real, args.exports_dir)
    fps, miss_fp = attach_case_dirs(fps, args.exports_dir)
    fns, miss_fn = attach_case_dirs(fns, args.exports_dir)

    missing_case_dirs = miss_h + miss_w + miss_nf + miss_nr + miss_fp + miss_fn

    print("[*] 可用候选池统计：")
    print(f"    High_Fake      : {len(high_fake)}")
    print(f"    Weak_Fake      : {len(weak_fake)}")
    print(f"    NoEvidence_Fake: {len(noe_fake)}")
    print(f"    NoEvidence_Real: {len(noe_real)}")
    print(f"    False_Positive : {len(fps)}")
    print(f"    False_Negative : {len(fns)}")

    selected = []
    selected_names = set()

    def add_bucket(bucket_name, candidates, n):
        nonlocal selected, selected_names
        candidates = [c for c in candidates if c["case_name"] not in selected_names]
        chosen = diverse_select(candidates, n, seed=args.seed + len(selected))
        for c in chosen:
            c = dict(c)
            c["protocol_bucket"] = bucket_name
            selected.append(c)
            selected_names.add(c["case_name"])
        return len(chosen)

    # 主采样：先严格按目标桶采
    count_by_bucket = {}
    count_by_bucket["High_Fake"] = add_bucket("High_Fake", high_fake, args.n_high_fake)
    count_by_bucket["Weak_Fake"] = add_bucket("Weak_Fake", weak_fake, args.n_weak_fake)
    count_by_bucket["NoEvidence_Fake"] = add_bucket("NoEvidence_Fake", noe_fake, args.n_noe_fake)
    count_by_bucket["NoEvidence_Real"] = add_bucket("NoEvidence_Real", noe_real, args.n_noe_real)
    count_by_bucket["False_Positive"] = add_bucket("False_Positive", fps, args.n_fp)
    count_by_bucket["False_Negative"] = add_bucket("False_Negative", fns, args.n_fn)

    # 如果不足 min_total，则从备用池补位
    # 补位优先级：
    # 1) 剩余 FP
    # 2) 剩余 FN
    # 3) 剩余 NoEvidence_Fake
    # 4) 剩余 NoEvidence_Real
    # 5) 剩余 Weak_Fake
    # 6) 剩余 High_Fake
    reserve_pool = []
    reserve_order = [
        ("False_Positive_Extra", fps),
        ("False_Negative_Extra", fns),
        ("NoEvidence_Fake_Extra", noe_fake),
        ("NoEvidence_Real_Extra", noe_real),
        ("Weak_Fake_Extra", weak_fake),
        ("High_Fake_Extra", high_fake),
    ]

    for bucket_name, pool in reserve_order:
        remain = [r for r in pool if r["case_name"] not in selected_names]
        extra = diverse_select(remain, args.max_total, seed=args.seed + 999)
        for r in extra:
            rr = dict(r)
            rr["protocol_bucket"] = bucket_name
            reserve_pool.append(rr)

    reserve_pool = dedupe_by_case(reserve_pool)

    while len(selected) < args.min_total and reserve_pool:
        r = reserve_pool.pop(0)
        if r["case_name"] in selected_names:
            continue
        selected.append(r)
        selected_names.add(r["case_name"])

    # 如果还没到 max_total，可以再少量补充到 max_total
    while len(selected) < args.max_total and reserve_pool:
        r = reserve_pool.pop(0)
        if r["case_name"] in selected_names:
            continue
        selected.append(r)
        selected_names.add(r["case_name"])

    # 建目录链接
    out_cases_dir = os.path.join(args.output_dir, "cases")
    os.makedirs(out_cases_dir, exist_ok=True)

    manifest_rows = []
    for r in selected:
        case_name = r["case_name"]
        original_case_dir = r["original_case_dir"]
        linked_case_dir = os.path.join(out_cases_dir, case_name)

        safe_symlink(original_case_dir, linked_case_dir)

        manifest_rows.append({
            "protocol_bucket": r["protocol_bucket"],
            "case_name": case_name,
            "gt_label": r["gt_label"],
            "pred_label": r["pred_label"],
            "correct": int(r["correct_bool"]),
            "status": r.get("status", ""),
            "dataset": r.get("dataset", ""),
            "source_type": r.get("source_type", ""),
            "pred_prob_fake": r.get("pred_prob_fake", ""),
            "linked_case_dir": linked_case_dir,
            "original_case_dir": original_case_dir
        })

    manifest_path = os.path.join(args.output_dir, "selected_manifest.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")

    if manifest_rows:
        write_csv(manifest_path, manifest_rows, list(manifest_rows[0].keys()))

    summary = {
        "requested": {
            "High_Fake": args.n_high_fake,
            "Weak_Fake": args.n_weak_fake,
            "NoEvidence_Fake": args.n_noe_fake,
            "NoEvidence_Real": args.n_noe_real,
            "False_Positive": args.n_fp,
            "False_Negative": args.n_fn,
            "min_total": args.min_total,
            "max_total": args.max_total
        },
        "selected_count": len(manifest_rows),
        "available_pool_count": {
            "High_Fake": len(high_fake),
            "Weak_Fake": len(weak_fake),
            "NoEvidence_Fake": len(noe_fake),
            "NoEvidence_Real": len(noe_real),
            "False_Positive": len(fps),
            "False_Negative": len(fns)
        },
        "selected_by_bucket": {
            k: sum(1 for r in manifest_rows if r["protocol_bucket"] == k)
            for k in sorted(set(r["protocol_bucket"] for r in manifest_rows))
        },
        "primary_bucket_count": count_by_bucket,
        "missing_case_dirs": missing_case_dirs
    }
    write_json(summary_path, summary)

    print("[+] 已生成 16–18 张完整版协议调试集")
    print(f"[*] manifest: {manifest_path}")
    print(f"[*] summary : {summary_path}")
    print(f"[*] case dir : {out_cases_dir}")


if __name__ == "__main__":
    main()