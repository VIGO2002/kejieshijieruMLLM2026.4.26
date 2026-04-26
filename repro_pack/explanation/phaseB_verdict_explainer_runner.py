import os
import re
import csv
import json
import glob
import argparse
from pathlib import Path

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError(
        "[!] 未安装 openai 包，请先执行: pip install openai\n"
        f"原始错误: {e}"
    )


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_json_from_text(text):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("[!] 无法从模型输出中解析合法 JSON。")


def normalize_label(x):
    s = str(x).strip().lower()
    if s == "fake":
        return "FAKE"
    if s == "real":
        return "REAL"
    return str(x).upper().strip()


def safe_float(x, default=0.0):
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def pick_router_pred_row(case_name, router_pred_rows):
    for row in router_pred_rows:
        if row.get("case_name", "") == case_name:
            return row
    return None


def build_router_decision_packet(router_pred_row):
    pred_label = normalize_label(router_pred_row.get("pred_label", "UNKNOWN"))
    pred_prob_fake = safe_float(router_pred_row.get("pred_prob_fake", 0.5), 0.5)

    if pred_label == "FAKE":
        fusion_confidence = pred_prob_fake
    elif pred_label == "REAL":
        fusion_confidence = 1.0 - pred_prob_fake
    else:
        fusion_confidence = 0.5

    return {
        "fusion_pred_label": pred_label,
        "fusion_confidence": round(float(fusion_confidence), 6)
    }


def build_router_basis_packet(fusion_row):
    """
    这里不直接读取 router 内部树结构解释，而是用当前主链中最可解释的上层信号
    构造一个稳定的 Router Basis Packet（路由器依据包）。
    """

    status = str(fusion_row.get("status", "")).strip()
    num_validated_objects = int(float(fusion_row.get("num_validated_objects", 0) or 0))
    evidence_coverage = safe_float(fusion_row.get("evidence_coverage", 0.0), 0.0)
    best_score_max = safe_float(fusion_row.get("best_score_max", 0.0), 0.0)
    fake_prob = safe_float(fusion_row.get("fake_prob", 0.0), 0.0)
    noe_score = safe_float(fusion_row.get("noevidence_fake_score", 0.0), 0.0)
    gap = safe_float(fusion_row.get("sys1_global_anomaly_gap", 0.0), 0.0)

    # ---- dominant_mode 判定 ----
    if status == "HighConfidenceEvidence" and num_validated_objects > 0 and evidence_coverage > 0:
        dominant_mode = "localized_evidence_dominant"
        top_supporting_signals = [
            {"name": "best_score_max", "value": round(best_score_max, 6)},
            {"name": "validated_evidence_objects", "value": num_validated_objects},
            {"name": "evidence_coverage", "value": round(evidence_coverage, 6)}
        ]
    elif status == "NoEvidence" or num_validated_objects == 0:
        dominant_mode = "global_anomaly_dominant"
        top_supporting_signals = [
            {"name": "noevidence_fake_score", "value": round(noe_score, 6)},
            {"name": "fake_prob", "value": round(fake_prob, 6)},
            {"name": "sys1_global_anomaly_gap", "value": round(gap, 6)}
        ]
    else:
        dominant_mode = "mixed"
        top_supporting_signals = [
            {"name": "best_score_max", "value": round(best_score_max, 6)},
            {"name": "noevidence_fake_score", "value": round(noe_score, 6)},
            {"name": "fake_prob", "value": round(fake_prob, 6)}
        ]

    return {
        "dominant_mode": dominant_mode,
        "top_supporting_signals": top_supporting_signals
    }


def render_phaseB_user_prompt(
    user_prompt_template,
    phaseA_output,
    blind_forensic_json,
    router_decision_packet,
    router_basis_packet,
    output_schema
):
    prompt = user_prompt_template
    prompt += "\n\n[Required Output JSON Schema]\n"
    prompt += json.dumps(output_schema, indent=2, ensure_ascii=False)

    prompt += "\n\n[Blind Auditor Output]\n"
    prompt += json.dumps(phaseA_output, indent=2, ensure_ascii=False)

    prompt += "\n\n[Blind Forensic JSON]\n"
    prompt += json.dumps(blind_forensic_json, indent=2, ensure_ascii=False)

    prompt += "\n\n[Router Decision Packet]\n"
    prompt += json.dumps(router_decision_packet, indent=2, ensure_ascii=False)

    prompt += "\n\n[Router Basis Packet]\n"
    prompt += json.dumps(router_basis_packet, indent=2, ensure_ascii=False)

    return prompt


def main():
    parser = argparse.ArgumentParser(description="Phase B Verdict Explainer Runner")
    parser.add_argument("--manifest", type=str, default="protocol_debug_set/selected_manifest.csv")
    parser.add_argument("--fusion_table", type=str, default="fusion_training_table_rf_core10_holdout.csv")
    parser.add_argument("--router_pred_csv", type=str, default="fusion_router_rf_core10_holdout_predictions.csv")
    parser.add_argument("--phaseA_dir", type=str, default="phaseA_protocol_debug_outputs")
    parser.add_argument("--protocol_dir", type=str, default="system2_protocol")
    parser.add_argument("--output_dir", type=str, default="phaseB_protocol_debug_outputs")
    parser.add_argument("--model", type=str, default="qwen3.6-flash")
    parser.add_argument("--api_key_env", type=str, default="DASHSCOPE_API_KEY")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key and (not args.dry_run):
        raise RuntimeError(f"[!] 环境变量 {args.api_key_env} 未设置。")

    # ---- load protocol files ----
    phaseB_system_prompt = read_text(os.path.join(args.protocol_dir, "phaseB_system_prompt.txt"))
    phaseB_user_prompt = read_text(os.path.join(args.protocol_dir, "phaseB_user_prompt.txt"))
    phaseB_output_schema = read_json(os.path.join(args.protocol_dir, "phaseB_output_schema.json"))

    # 这两个模板当前不强依赖，但保留读取，便于后续协议版本管理
    router_decision_template_path = os.path.join(args.protocol_dir, "router_decision_packet_template.json")
    router_basis_template_path = os.path.join(args.protocol_dir, "router_basis_packet_template.json")
    if os.path.exists(router_decision_template_path):
        _ = read_json(router_decision_template_path)
    if os.path.exists(router_basis_template_path):
        _ = read_json(router_basis_template_path)

    manifest_rows = read_csv_rows(args.manifest)
    fusion_rows = read_csv_rows(args.fusion_table)
    router_pred_rows = read_csv_rows(args.router_pred_csv)

    fusion_by_case = {r["case_name"]: r for r in fusion_rows}

    client = None
    if not args.dry_run:
        client = OpenAI(
            api_key=api_key,
            base_url=args.base_url
        )

    for idx, row in enumerate(manifest_rows, start=1):
        case_name = row["case_name"]

        if case_name not in fusion_by_case:
            print(f"[!] [{idx}] 跳过 {case_name}：fusion_table 中找不到")
            continue

        fusion_row = fusion_by_case[case_name]
        router_pred_row = pick_router_pred_row(case_name, router_pred_rows)
        if router_pred_row is None:
            print(f"[!] [{idx}] 跳过 {case_name}：router_pred_csv 中找不到")
            continue

        phaseA_case_dir = os.path.join(args.phaseA_dir, case_name)
        phaseA_output_path = os.path.join(phaseA_case_dir, "phaseA_output.json")
        blind_forensic_path = os.path.join(phaseA_case_dir, "phaseA_input_blind_forensic.json")

        if not os.path.exists(phaseA_output_path):
            print(f"[!] [{idx}] 跳过 {case_name}：缺少 phaseA_output.json")
            continue
        if not os.path.exists(blind_forensic_path):
            print(f"[!] [{idx}] 跳过 {case_name}：缺少 phaseA_input_blind_forensic.json")
            continue

        phaseA_output = read_json(phaseA_output_path)
        blind_forensic_json = read_json(blind_forensic_path)
        router_decision_packet = build_router_decision_packet(router_pred_row)
        router_basis_packet = build_router_basis_packet(fusion_row)

        rendered_user_prompt = render_phaseB_user_prompt(
            user_prompt_template=phaseB_user_prompt,
            phaseA_output=phaseA_output,
            blind_forensic_json=blind_forensic_json,
            router_decision_packet=router_decision_packet,
            router_basis_packet=router_basis_packet,
            output_schema=phaseB_output_schema
        )

        case_out_dir = os.path.join(args.output_dir, case_name)
        os.makedirs(case_out_dir, exist_ok=True)

        phaseB_input_bundle = {
            "blind_audit_output": phaseA_output,
            "blind_forensic_json": blind_forensic_json,
            "router_decision_packet": router_decision_packet,
            "router_basis_packet": router_basis_packet
        }

        write_json(os.path.join(case_out_dir, "phaseB_input_bundle.json"), phaseB_input_bundle)
        write_json(os.path.join(case_out_dir, "router_decision_packet.json"), router_decision_packet)
        write_json(os.path.join(case_out_dir, "router_basis_packet.json"), router_basis_packet)
        write_text(os.path.join(case_out_dir, "phaseB_rendered_user_prompt.txt"), rendered_user_prompt)
        write_text(os.path.join(case_out_dir, "phaseB_system_prompt.txt"), phaseB_system_prompt)

        print(
            f"[*] [{idx}/{len(manifest_rows)}] case={case_name} | "
            f"router={router_decision_packet['fusion_pred_label']} "
            f"({router_decision_packet['fusion_confidence']:.4f}) | "
            f"mode={router_basis_packet['dominant_mode']}"
        )

        if args.dry_run:
            continue

        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": phaseB_system_prompt},
                    {"role": "user", "content": rendered_user_prompt}
                ],
                temperature=0.1
            )

            raw_text = resp.choices[0].message.content
            write_text(os.path.join(case_out_dir, "phaseB_output_raw.txt"), raw_text)

            parsed = parse_json_from_text(raw_text)
            write_json(os.path.join(case_out_dir, "phaseB_output.json"), parsed)

        except Exception as e:
            write_text(os.path.join(case_out_dir, "phaseB_error.txt"), str(e))
            print(f"[!] Phase B 调用失败: {case_name} | {e}")

    print("\n[+] Phase B Runner 完成")


if __name__ == "__main__":
    main()