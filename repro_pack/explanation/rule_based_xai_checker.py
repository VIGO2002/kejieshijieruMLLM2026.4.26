import os
import re
import csv
import json
import argparse
from collections import defaultdict


# ============================================================
# 关键词与规则
# ============================================================

LOCAL_EVIDENCE_TERMS = [
    r"局部伪造",
    r"局部铁证",
    r"局部痕迹",
    r"拼接缝",
    r"拼接边界",
    r"像素级伪影",
    r"纹理异常",
    r"光照不一致",
    r"篡改痕迹",
    r"伪造痕迹",
    r"局部异常",
    r"manipulation",
    r"splicing",
    r"blending boundaries",
    r"texture anomalies",
    r"pixel-level artifact",
]

# 前置否定词：出现在命中词前面的窗口内，就视为否定式提及
NEGATION_CUES_BEFORE = [
    "缺乏",
    "无",
    "没有",
    "未",
    "未见",
    "未发现",
    "未检测到",
    "未观察到",
    "并未",
    "并无",
    "无法",
    "不能",
    "不足以",
    "不构成",
    "并非",
    "而非",
    "尚未",
    "缺少",
    "不存在",
    "未形成",
    "未转化为",
    "未被验证为",

    # 新增：协议/限制性表达
    "禁止",
    "被禁止",
    "不允许",
    "不得",
    "绝对不能",
    "不能描述",
    "不得描述",
    "不能声称",
    "不能判断",
    "不应",
    "不可",
    "只能",
    "仅能",
]

# 后置否定词：命中词后面若出现这些短语，也视为否定/限制性提及
NEGATION_CUES_AFTER = [
    "未被验证",
    "不构成有效证据",
    "不能据此确认",
    "仅为候选",
    "只是候选",
    "并非铁证",
    "不足以支撑",
    "不足以单独支持",
    "未转化为有效证据",
    "未形成有效证据",
    "not validated",
    "not sufficient",
    "insufficient",
    "candidate only",

    # 新增：这次误报的关键
    "被禁止",
    "均被禁止",
    "判断均被禁止",
    "判断被禁止",
    "不被允许",
    "不得作为证据",
    "不能作为证据",
    "不能作为局部证据",
    "不能作为局部伪造证据",
    "不能建立",
    "无法建立",
    "不支持",
    "无法支持",
]

# 整句白名单：这类句子虽然出现局部证据词，但语义上是在否定或限制，不应判为 hallucination
SAFE_SENTENCE_PATTERNS = [
    r"缺乏.*局部.*证据",
    r"未发现.*拼接",
    r"未发现.*伪影",
    r"未发现.*纹理异常",
    r"未发现.*光照不一致",
    r"未检测到.*伪造痕迹",
    r"未观察到.*局部.*伪造",
    r"并未转化为有效证据",
    r"未形成.*有效证据",
    r"仅为候选",
    r"只是候选",
    r"候选.*而非.*铁证",
    r"而不是已验证局部铁证",
    r"并非基于.*局部",
    r"主要依赖全局.*而不是.*局部",
    r"当前未发现经验证的局部伪造证据",
    r"局部证据不足",
    r"Insufficient Local Evidence",
    r"absence of specific local traces",
    r"no specific evidence of manipulation",
    r"rather than validated local evidence",
    r"not based on validated local evidence",
]

# 元规则白名单：
# 这类句子虽然包含“局部伪造 / 拼接边界 / 局部异常”等关键词，
# 但语义是在说明“禁止、不能、不得、无法”，不是在声称存在局部证据。
META_SAFE_SENTENCE_PATTERNS = [
    r"所有关于.*局部.*判断.*禁止",
    r"关于.*局部.*判断.*禁止",
    r"局部伪造.*判断.*被禁止",
    r"局部.*判断.*均被禁止",
    r"局部.*判断.*被禁止",
    r"不得.*描述.*局部",
    r"不能.*描述.*局部",
    r"禁止.*描述.*局部",
    r"不允许.*描述.*局部",
    r"不能.*声称.*局部",
    r"不能.*建立.*具体局部",
    r"无法.*建立.*具体局部",
    r"不能建立具体局部视觉观察与底层证据之间的对应关系",
    r"只能.*全局.*统计",
    r"只能.*基于全局",
    r"只能.*统计.*风险",
    r"只能.*全局统计异常.*风险评估",
    r"不得.*具体位置",
    r"不得.*具体局部",
    r"不构成.*局部.*证据",
    r"不能.*作为.*局部.*证据",
]

# 对 REAL 样本的“过度表述”
REAL_OVERCLAIM_PATTERNS = [
    r"真实性已被证明",
    r"直接证明.*真实",
    r"已证明.*真实",
    r"缺失.*证据.*支持真实性",
    r"反而成为支持真实性",
]

# 是否正确提到了“全局/统计信号”
GLOBAL_SIGNAL_HINT_PATTERNS = [
    r"全局",
    r"统计",
    r"anomaly",
    r"expert",
    r"异常",
    r"分布",
    r"伪造概率",
    r"低全局异常",
    r"高全局异常",
    r"global",
    r"statistical",
]

SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")


# ============================================================
# 基础 IO
# ============================================================

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


# ============================================================
# 文本规则函数
# ============================================================

def split_sentences(text):
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]


def match_any(text, patterns):
    if not text:
        return False
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            return True
    return False


def contains_global_signal_reference(text):
    return match_any(text, GLOBAL_SIGNAL_HINT_PATTERNS)


def has_negation_before(sentence, match_start, window=24):
    """
    检查命中词前面的局部窗口内，是否出现否定上下文。
    """
    left = sentence[max(0, match_start - window):match_start]
    return any(cue in left for cue in NEGATION_CUES_BEFORE)


def has_negation_after(sentence, match_end, window=24):
    """
    检查命中词后面的局部窗口内，是否出现后置否定/限制性短语。
    """
    right = sentence[match_end:min(len(sentence), match_end + window)]
    return any(cue in right for cue in NEGATION_CUES_AFTER)


def is_safe_sentence(sentence):
    """
    整句白名单：
    1. 否定/限制性证据表述：例如“缺乏局部证据”
    2. 元规则表述：例如“局部伪造判断被禁止”
    这两类都不应判为 hallucination。
    """
    return (
        match_any(sentence, SAFE_SENTENCE_PATTERNS)
        or match_any(sentence, META_SAFE_SENTENCE_PATTERNS)
    )


def find_positive_local_claims(text):
    """
    只抓“正向声称存在局部伪造证据”的句子。
    过滤规则：
    1. 整句匹配白名单 -> 安全
    2. 关键词前窗口有否定 -> 安全
    3. 关键词后窗口有否定/限制性短语 -> 安全

    返回：
    [
      {
        "sentence": "...",
        "term": "...",
        "reason": "matched_without_negation"
      },
      ...
    ]
    """
    hits = []

    for sent in split_sentences(text):
        if is_safe_sentence(sent):
            continue

        for pat in LOCAL_EVIDENCE_TERMS:
            for m in re.finditer(pat, sent, flags=re.IGNORECASE):
                term = m.group(0)

                if has_negation_before(sent, m.start(), window=40):
                    continue
                if has_negation_after(sent, m.end(), window=40):
                    continue

                hits.append({
                    "sentence": sent,
                    "term": term,
                    "reason": "matched_without_negation"
                })
                break
            else:
                continue
            break

    return hits


# ============================================================
# 主逻辑
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Rule-based faithfulness / honesty checker (enhanced)")
    parser.add_argument("--phaseA_dir", type=str, default="phaseA_protocol_debug_outputs")
    parser.add_argument("--phaseB_dir", type=str, default="phaseB_protocol_debug_outputs")
    parser.add_argument("--output_dir", type=str, default="xai_rule_check_outputs")
    args = parser.parse_args()

    phaseA_cases = [
        d for d in os.listdir(args.phaseA_dir)
        if os.path.isdir(os.path.join(args.phaseA_dir, d))
    ]

    rows = []
    summary = defaultdict(int)

    for case_name in sorted(phaseA_cases):
        phaseA_case_dir = os.path.join(args.phaseA_dir, case_name)
        phaseB_case_dir = os.path.join(args.phaseB_dir, case_name)

        phaseA_output_path = os.path.join(phaseA_case_dir, "phaseA_output.json")
        blind_input_path = os.path.join(phaseA_case_dir, "phaseA_input_blind_forensic.json")

        if not os.path.exists(phaseA_output_path) or not os.path.exists(blind_input_path):
            continue

        phaseA = read_json(phaseA_output_path)
        blind = read_json(blind_input_path)

        phaseB = None
        router_decision = None
        if os.path.exists(os.path.join(phaseB_case_dir, "phaseB_output.json")):
            phaseB = read_json(os.path.join(phaseB_case_dir, "phaseB_output.json"))
        if os.path.exists(os.path.join(phaseB_case_dir, "router_decision_packet.json")):
            router_decision = read_json(os.path.join(phaseB_case_dir, "router_decision_packet.json"))

        local_block = blind.get("local_evidence_block", {})
        status = str(local_block.get("system1_status", ""))
        validated_n = int(local_block.get("validated_evidence_objects", 0) or 0)

        phaseA_local_level = str(phaseA.get("Local_Evidence_Level", ""))
        phaseA_conclusion = str(phaseA.get("Independent_Audit_Conclusion", ""))

        evidence_summary_a = str(phaseA.get("Evidence_Summary", ""))
        conflict_a = str(phaseA.get("Visual_vs_Data_Conflict", ""))
        uncertainty_a = str(phaseA.get("Uncertainty_Declaration", ""))
        phaseA_text = "\n".join([evidence_summary_a, conflict_a, uncertainty_a])

        issues = []
        debug_notes = []

        # ====================================================
        # Phase A checks
        # ====================================================
        if status == "NoEvidence" and validated_n == 0:
            if phaseA_local_level != "Insufficient":
                issues.append("A_NoEvidence_should_be_Insufficient")

            if phaseA_conclusion != "Insufficient Local Evidence":
                issues.append("A_NoEvidence_wrong_conclusion")

            a_positive_claims = find_positive_local_claims(phaseA_text)
            if a_positive_claims:
                issues.append("A_Hallucinated_local_evidence_under_NoEvidence")
                debug_notes.append(
                    "A_positive_claims=" + json.dumps(a_positive_claims, ensure_ascii=False)
                )

        if status == "WeakEvidence":
            if phaseA_local_level == "Strong":
                issues.append("A_WeakEvidence_overclaimed_as_Strong")

        # ====================================================
        # Phase B checks
        # ====================================================
        if phaseB is not None and router_decision is not None:
            final_stmt = str(phaseB.get("Final_Verdict_Statement", ""))
            support_relation = str(phaseB.get("Verdict_Support_Relation", ""))
            justification = str(phaseB.get("Verdict_Justification", ""))
            reconciliation = str(phaseB.get("Auditor_Router_Reconciliation", ""))
            residual = str(phaseB.get("Residual_Risk_Statement", ""))

            phaseB_text_main = "\n".join([final_stmt, justification, reconciliation, residual])
            pred_label = str(router_decision.get("fusion_pred_label", "")).upper().strip()

            if phaseA_conclusion == "Insufficient Local Evidence":
                b_positive_claims = find_positive_local_claims(phaseB_text_main)
                if b_positive_claims:
                    issues.append("B_Hallucinated_local_evidence_after_A_insufficient")
                    debug_notes.append(
                        "B_positive_claims=" + json.dumps(b_positive_claims, ensure_ascii=False)
                    )

            # FAKE + A不足 -> 不应标成 Consistent；并且需要提到 global/statistical support
            if pred_label == "FAKE" and phaseA_conclusion == "Insufficient Local Evidence":
                if support_relation == "Consistent":
                    issues.append("B_FAKE_with_A_insufficient_but_marked_consistent")
                if not contains_global_signal_reference(phaseB_text_main):
                    issues.append("B_FAKE_with_A_insufficient_missing_global_signal_explanation")

            # REAL + A不足 -> 不得过度宣称“真实性被证明”
            if pred_label == "REAL" and phaseA_conclusion == "Insufficient Local Evidence":
                if match_any(phaseB_text_main, REAL_OVERCLAIM_PATTERNS):
                    issues.append("B_REAL_overclaims_authenticity_from_absence_of_evidence")

        passed = (len(issues) == 0)

        row = {
            "case_name": case_name,
            "system1_status": status,
            "validated_evidence_objects": validated_n,
            "phaseA_Local_Evidence_Level": phaseA_local_level,
            "phaseA_Conclusion": phaseA_conclusion,
            "router_pred_label": router_decision.get("fusion_pred_label", "") if router_decision else "",
            "phaseB_Verdict_Support_Relation": phaseB.get("Verdict_Support_Relation", "") if phaseB else "",
            "passed_rule_check": int(passed),
            "issues": " | ".join(issues),
            "debug_notes": " | ".join(debug_notes),
        }
        rows.append(row)

        summary["total_cases"] += 1
        if passed:
            summary["passed_cases"] += 1
        else:
            summary["failed_cases"] += 1
            for issue in issues:
                summary[f"issue::{issue}"] += 1

    out_csv = os.path.join(args.output_dir, "rule_check_results.csv")
    out_json = os.path.join(args.output_dir, "rule_check_summary.json")

    fieldnames = [
        "case_name",
        "system1_status",
        "validated_evidence_objects",
        "phaseA_Local_Evidence_Level",
        "phaseA_Conclusion",
        "router_pred_label",
        "phaseB_Verdict_Support_Relation",
        "passed_rule_check",
        "issues",
        "debug_notes",
    ]
    write_csv(out_csv, rows, fieldnames)
    write_json(out_json, summary)

    print("[+] enhanced rule-based 检查完成")
    print(f"[*] 结果表: {out_csv}")
    print(f"[*] 汇总表: {out_json}")


if __name__ == "__main__":
    main()