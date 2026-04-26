import os
import re
import io
import csv
import json
import glob
import base64
import argparse
from pathlib import Path

# ===== OpenAI-compatible client（兼容 OpenAI 的客户端）=====
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


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def find_first(patterns):
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None


def list_crops(crops_dir, max_crops=2):
    if not os.path.isdir(crops_dir):
        return []
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.WEBP")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(crops_dir, ext)))
    files = sorted(files)
    return files[:max_crops]


def image_to_data_url(path):
    suffix = Path(path).suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp"
    }.get(suffix, "image/png")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def parse_json_from_text(text):
    text = text.strip()

    # 先尝试整段直接解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 再尝试抓取第一个 JSON 对象
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("[!] 无法从模型输出中解析合法 JSON。")


def build_blind_forensic_json(case_name, fusion_row, evidence_json, has_crops, crop_count):
    # -------- local evidence block --------
    system1_status = (
        evidence_json.get("final_system1_status")
        or evidence_json.get("system1_status")
        or fusion_row.get("status", "")
    )

    validated_objs = evidence_json.get("validated_evidence_objects", [])
    if isinstance(validated_objs, list):
        num_validated = len(validated_objs)
    else:
        # 兼容旧格式
        try:
            num_validated = int(validated_objs)
            validated_objs = []
        except Exception:
            num_validated = int(float(fusion_row.get("num_validated_objects", 0)))
            validated_objs = []

    details = []
    if isinstance(validated_objs, list):
        for i, obj in enumerate(validated_objs[:2], start=1):
            bbox = obj.get("bbox", obj.get("bbox_xyxy", []))
            details.append({
                "region_id": obj.get("region_id", i),
                "support_level": obj.get("support_level", obj.get("evidence_level", "unknown")),
                "bbox": bbox if isinstance(bbox, list) else [],
                "score_max": obj.get("score_max", obj.get("score", fusion_row.get("best_score_max", 0.0))),
                "token_contrast": obj.get("token_contrast", fusion_row.get("best_token_contrast", 0.0)),
                "note": obj.get("note", "")
            })

    # NoEvidence 强制不给 detail
    if str(system1_status) == "NoEvidence":
        details = []
        num_validated = 0
        has_crops = False
        crop_count = 0

    blind_json = {
        "case_name": case_name,
        "image_meta": {
            "crops_provided": bool(has_crops),
            "crop_count": int(crop_count)
        },
        "local_evidence_block": {
            "system1_status": system1_status,
            "validated_evidence_objects": int(num_validated),
            "best_score_max": float(fusion_row.get("best_score_max", 0.0)),
            "best_token_contrast": float(fusion_row.get("best_token_contrast", 0.0)),
            "evidence_coverage": float(fusion_row.get("evidence_coverage", 0.0)),
            "validated_evidence_objects_detail": details
        },
        "global_signal_block": {
            "fake_prob": float(fusion_row.get("fake_prob", 0.0)),
            "sys1_global_anomaly_prefix": float(fusion_row.get("sys1_global_anomaly_prefix", 0.0)),
            "sys1_global_anomaly_gap": float(fusion_row.get("sys1_global_anomaly_gap", 0.0)),
            "noevidence_expert_score": float(fusion_row.get("noevidence_fake_score", 0.0))
        }
    }
    return blind_json


def render_phaseA_user_prompt(user_prompt_template, blind_json, output_schema):
    prompt = user_prompt_template
    prompt += "\n\n[Required Output JSON Schema]\n"
    prompt += json.dumps(output_schema, indent=2, ensure_ascii=False)
    prompt += "\n\n[Blind Forensic JSON]\n"
    prompt += json.dumps(blind_json, indent=2, ensure_ascii=False)
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Phase A Blind Auditor Runner")
    parser.add_argument("--manifest", type=str, default="protocol_debug_set/selected_manifest.csv")
    parser.add_argument("--fusion_table", type=str, default="fusion_training_table_rf_core10_holdout.csv")
    parser.add_argument("--protocol_dir", type=str, default="system2_protocol")
    parser.add_argument("--output_dir", type=str, default="phaseA_protocol_debug_outputs")
    parser.add_argument("--model", type=str, default="qwen3.5-flash")
    parser.add_argument("--api_key_env", type=str, default="DASHSCOPE_API_KEY")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--max_crops", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key and (not args.dry_run):
        raise RuntimeError(
            f"[!] 环境变量 {args.api_key_env} 未设置。"
        )

    # ---- load protocol files ----
    phaseA_system_prompt = read_text(os.path.join(args.protocol_dir, "phaseA_system_prompt.txt"))
    phaseA_user_prompt = read_text(os.path.join(args.protocol_dir, "phaseA_user_prompt.txt"))
    phaseA_output_schema = read_json(os.path.join(args.protocol_dir, "phaseA_output_schema.json"))

    # 这里只是为了协议完整性；当前 runner 直接程序构造 blind json
    blind_template_path = os.path.join(args.protocol_dir, "blind_forensic_json_template.json")
    if os.path.exists(blind_template_path):
        _ = read_json(blind_template_path)

    fusion_rows = read_csv_rows(args.fusion_table)
    fusion_by_case = {r["case_name"]: r for r in fusion_rows}

    manifest_rows = read_csv_rows(args.manifest)

    client = None
    if not args.dry_run:
        client = OpenAI(
            api_key=api_key,
            base_url=args.base_url
        )

    for idx, row in enumerate(manifest_rows, start=1):
        case_name = row["case_name"]
        case_dir = row["original_case_dir"]

        if case_name not in fusion_by_case:
            print(f"[!] [{idx}] 跳过 {case_name}：fusion_table 中找不到")
            continue

        fusion_row = fusion_by_case[case_name]

        # ---- locate assets ----
        image_raw = find_first([
            os.path.join(case_dir, "*_original_raw.png"),
            os.path.join(case_dir, "*_original_224.png"),
        ])
        image_overlay = find_first([
            os.path.join(case_dir, "*_overlay_raw.png"),
            os.path.join(case_dir, "*_overlay_224.png"),
        ])
        evidence_full = find_first([
            os.path.join(case_dir, "*_evidence_full.json"),
            os.path.join(case_dir, "*_evidence.json"),
            os.path.join(case_dir, "*_evidence_for_model.json"),
        ])
        crops = list_crops(os.path.join(case_dir, "crops_raw"), max_crops=args.max_crops)

        if image_raw is None or image_overlay is None or evidence_full is None:
            print(f"[!] [{idx}] 跳过 {case_name}：缺少原图/overlay/evidence_full")
            continue

        evidence_json = read_json(evidence_full)

        system1_status = evidence_json.get("final_system1_status", fusion_row.get("status", ""))
        if str(system1_status) == "NoEvidence":
            crops = []

        blind_json = build_blind_forensic_json(
            case_name=case_name,
            fusion_row=fusion_row,
            evidence_json=evidence_json,
            has_crops=(len(crops) > 0),
            crop_count=len(crops)
        )

        rendered_user_prompt = render_phaseA_user_prompt(
            phaseA_user_prompt,
            blind_json,
            phaseA_output_schema
        )

        case_out_dir = os.path.join(args.output_dir, case_name)
        os.makedirs(case_out_dir, exist_ok=True)

        write_json(os.path.join(case_out_dir, "phaseA_input_blind_forensic.json"), blind_json)
        write_text(os.path.join(case_out_dir, "phaseA_rendered_user_prompt.txt"), rendered_user_prompt)
        write_text(os.path.join(case_out_dir, "phaseA_system_prompt.txt"), phaseA_system_prompt)

        print(f"[*] [{idx}/{len(manifest_rows)}] case={case_name} | status={system1_status} | crops={len(crops)}")

        if args.dry_run:
            continue

        content = [
            {"type": "text", "text": rendered_user_prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_raw)}},
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_overlay)}},
        ]
        for c in crops:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(c)}})

        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": phaseA_system_prompt},
                    {"role": "user", "content": content}
                ],
                temperature=0.1
            )

            raw_text = resp.choices[0].message.content
            write_text(os.path.join(case_out_dir, "phaseA_output_raw.txt"), raw_text)

            parsed = parse_json_from_text(raw_text)
            write_json(os.path.join(case_out_dir, "phaseA_output.json"), parsed)

        except Exception as e:
            write_text(os.path.join(case_out_dir, "phaseA_error.txt"), str(e))
            print(f"[!] Phase A 调用失败: {case_name} | {e}")

    print("\n[+] Phase A Runner 完成")


if __name__ == "__main__":
    main()