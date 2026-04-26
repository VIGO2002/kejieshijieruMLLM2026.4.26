def _get_float(result: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(result.get(key, default))
    except Exception:
        return default


def _get_int(result: dict, key: str, default: int = 0) -> int:
    try:
        return int(result.get(key, default))
    except Exception:
        return default


def build_explanation(result: dict) -> str:
    """
    根据 System 1 的结构化证据生成解释文字。

    注意：
    - 这里不调用 MLLM（多模态大语言模型）
    - 解释完全来自 fake_prob、system1_status、score/token/coverage 等结构化指标
    - 蓝色框只解释为候选区域，不解释为伪造铁证
    """

    pred_label = result.get("pred_label", "Unknown")
    binary_pred_label = result.get("binary_pred_label", "Unknown")

    # 兼容 system1_demo_infer.py 和旧版字段
    status = result.get("system1_status", result.get("status", "NoEvidence"))

    fake_prob = _get_float(result, "fake_prob", 0.0)
    best_score = _get_float(result, "best_score_max", 0.0)
    best_ratio = _get_float(result, "best_contrast_ratio", 0.0)
    best_token = _get_float(result, "best_token_contrast", 0.0)
    coverage = _get_float(result, "evidence_coverage", 0.0)

    num_raw = _get_int(result, "num_raw_clusters", 0)
    num_singletons = _get_int(result, "num_singletons", 0)
    num_valid = _get_int(result, "num_valid_clusters", 0)

    has_high_conf = bool(result.get("has_high_conf", False))
    has_any_passed = bool(result.get("has_any_passed", False))
    rescued_by_token = bool(result.get("rescued_by_token", False))

    # ------------------------------------------------------------
    # 1. 总体判断
    # ------------------------------------------------------------
    if pred_label == "Likely AI-generated":
        headline = "系统倾向于判断该图像为 AI 生成图像。"
    elif pred_label == "Likely real":
        headline = "系统倾向于判断该图像为真实图像。"
    elif pred_label == "Uncertain":
        headline = "系统认为该图像处于不确定区间，暂不建议给出强真伪结论。"
    else:
        if fake_prob >= 0.65:
            headline = "系统倾向于判断该图像为 AI 生成图像。"
        elif fake_prob <= 0.35:
            headline = "系统倾向于判断该图像为真实图像。"
        else:
            headline = "系统认为该图像处于不确定区间，暂不建议给出强真伪结论。"

    prob_text = (
        f"模型输出的 fake_prob 为 {fake_prob:.3f}。"
        f"在二分类阈值下，内部判断为 {binary_pred_label}。"
        "为了公开演示更稳健，界面采用三段式输出："
        "高 fake_prob 倾向于 AI 生成，低 fake_prob 倾向于真实，中间区间标记为不确定。"
    )

    # ------------------------------------------------------------
    # 2. 局部证据解释
    # ------------------------------------------------------------
    if status == "HighConfidenceEvidence":
        evidence_text = (
            "局部证据方面，System 1（系统1）检测到了高置信证据区域。"
            "这表示至少一个候选区域通过了局部证据验证，并且其局部统计对比强度较高。"
            "在可视化图中，红色框表示高置信证据区域。"
        )
    elif status == "WeakEvidence":
        evidence_text = (
            "局部证据方面，System 1 检测到了弱证据区域。"
            "这表示图像中存在一定异常线索，但局部证据强度不足以形成高置信判断。"
            "在可视化图中，橙色框表示弱证据区域。"
        )
    else:
        evidence_text = (
            "局部证据方面，System 1 没有检测到通过验证的证据区域。"
            "如果可视化图中出现蓝色框，它只表示模型提出的候选区域，"
            "不能被解释为已经验证通过的伪造证据。"
            "此时判断主要依赖全局 fake_prob，而不是明确的局部证据。"
        )

    # ------------------------------------------------------------
    # 3. 指标解释
    # ------------------------------------------------------------
    metric_text = (
        "结构化指标如下："
        f"num_raw_clusters={num_raw}，"
        f"num_singletons={num_singletons}，"
        f"num_valid_clusters={num_valid}，"
        f"best_score_max={best_score:.4f}，"
        f"best_contrast_ratio={best_ratio:.4f}，"
        f"best_token_contrast={best_token:.4f}，"
        f"evidence_coverage={coverage:.4f}。"
    )

    metric_meaning = (
        "其中，best_score_max 表示最强候选区域的 token-level suspiciousness scores"
        "（token 级可疑度分数）峰值；"
        "best_contrast_ratio 表示候选区域相对邻域的局部统计对比强度；"
        "best_token_contrast 表示候选区域相对周边 patch-token"
        "（图像块特征 token）的特征异常差异；"
        "evidence_coverage 表示通过验证的证据区域覆盖了多少 top suspicious patches"
        "（最高可疑图像块）。"
    )

    # ------------------------------------------------------------
    # 4. 额外状态说明
    # ------------------------------------------------------------
    flags = []

    if has_high_conf:
        flags.append("存在高置信局部证据。")
    elif has_any_passed:
        flags.append("存在通过验证的局部证据，但强度为弱证据。")
    else:
        flags.append("没有通过验证的局部证据。")

    if rescued_by_token:
        flags.append(
            "部分证据区域主要由 token_contrast 触发，说明该区域在特征空间中存在异常，"
            "但像素层面的局部统计对比不一定很强。"
        )

    flag_text = " ".join(flags)

    # ------------------------------------------------------------
    # 5. 风险提示
    # ------------------------------------------------------------
    caution = (
        "需要注意的是，该结果是模型辅助判断，不应被理解为绝对法医结论。"
        "对于强压缩、低分辨率、重度后处理、截图转发或明显超出训练分布的图像，"
        "模型判断可能不稳定。"
    )

    return "\n\n".join(
        [
            headline,
            prob_text,
            evidence_text,
            metric_text,
            metric_meaning,
            flag_text,
            caution,
        ]
    )