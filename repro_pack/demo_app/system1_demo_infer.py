import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms


# ============================================================
# 路径处理
# ============================================================
# 当前文件: repro_pack/demo_app/system1_demo_infer.py
# REPRO_ROOT: repro_pack
REPRO_ROOT = Path(__file__).resolve().parents[1]
FGTS_CORE_DIR = REPRO_ROOT / "fgts_core"

# 让 cross_domain_eval.py 可以 import test_system1_protocol.py
if str(FGTS_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(FGTS_CORE_DIR))


from cross_domain_eval import (  # noqa: E402
    SCORE_TH,
    RATIO_TH,
    TOKEN_TH,
    GLOBAL_FAKE_GATE,
    build_evidence_objects,
)

from test_system1_protocol import (  # noqa: E402
    DinoV3Model,
    LinearProbe,
    REAL_CLASS_IDX,
    FAKE_CLASS_IDX,
    tensor_to_gray_uint8,
    cluster_patch_indices,
)


# ============================================================
# 工具函数
# ============================================================
def to_python(obj: Any) -> Any:
    """把 tensor / numpy 类型递归转成 JSON 可序列化的 Python 基本类型。"""
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()

    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_python(x) for x in obj]

    if isinstance(obj, tuple):
        return [to_python(x) for x in obj]

    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def scale_bbox_224_to_raw(bbox_224: List[int], raw_size: Tuple[int, int]) -> List[int]:
    """把 224×224 坐标映射回原图尺寸。"""
    raw_w, raw_h = raw_size
    x1, y1, x2, y2 = bbox_224

    sx = raw_w / 224.0
    sy = raw_h / 224.0

    rx1 = int(round(x1 * sx))
    ry1 = int(round(y1 * sy))
    rx2 = int(round(x2 * sx))
    ry2 = int(round(y2 * sy))

    rx1 = max(0, min(rx1, raw_w - 1))
    ry1 = max(0, min(ry1, raw_h - 1))
    rx2 = max(1, min(rx2, raw_w))
    ry2 = max(1, min(ry2, raw_h))

    return [rx1, ry1, rx2, ry2]


def attach_raw_geometry(objects: List[Dict[str, Any]], raw_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    """给 proposal / validated evidence object 增加 bbox_raw。"""
    new_objects = []
    for obj in objects:
        new_obj = dict(obj)
        if "bbox_224" in new_obj:
            new_obj["bbox_raw"] = scale_bbox_224_to_raw(new_obj["bbox_224"], raw_size)
        new_objects.append(new_obj)
    return new_objects


def draw_overlay(
    image: Image.Image,
    validated_objects: List[Dict[str, Any]],
    proposal_objects: List[Dict[str, Any]],
) -> Image.Image:
    """
    在原图尺寸上画证据框：
    - 红色：HighConfidenceEvidence
    - 橙色：WeakEvidence
    - 蓝色：NoEvidence 时仅显示 top proposal，表示“候选但未验证通过”
    """
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)

    if len(validated_objects) > 0:
        for obj in validated_objects:
            bbox = obj.get("bbox_raw")
            if bbox is None:
                continue

            level = obj.get("evidence_level", "")
            rank = obj.get("rank", -1)

            if level == "HighConfidenceEvidence":
                color = (255, 0, 0)       # red
                width = 4
            else:
                color = (255, 165, 0)     # orange
                width = 4

            draw.rectangle(bbox, outline=color, width=width)

            x1, y1, _, _ = bbox
            tag_text = str(rank)
            tag_w, tag_h = 28, 18
            draw.rectangle([x1, max(0, y1 - tag_h), x1 + tag_w, y1], fill=color)
            draw.text((x1 + 6, max(0, y1 - tag_h + 2)), tag_text, fill=(0, 0, 0))

    else:
        # 没有验证通过的证据时，只画最强 proposal 作为参考，不能当成铁证
        if len(proposal_objects) > 0:
            top_obj = sorted(
                proposal_objects,
                key=lambda x: (
                    float(x.get("score_max", 0.0)),
                    float(x.get("token_contrast", 0.0)),
                    float(x.get("contrast_ratio", 0.0)),
                ),
                reverse=True,
            )[0]
            bbox = top_obj.get("bbox_raw")
            if bbox is not None:
                draw.rectangle(bbox, outline=(30, 144, 255), width=3)

    return overlay


# ============================================================
# 主检测器
# ============================================================
class System1DemoDetector:
    def __init__(
        self,
        device: str = None,
        model_name: str = "dinov3_vit_7b",
        ckpt_path: str = None,
        fake_high_threshold: float = 0.65,
        fake_low_threshold: float = 0.35,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.fake_high_threshold = fake_high_threshold
        self.fake_low_threshold = fake_low_threshold

        if ckpt_path is None:
            ckpt_path = REPRO_ROOT / "checkpoints" / "AIGCDetectionBenchmark" / "linear_probe.pth"
        else:
            ckpt_path = Path(ckpt_path)

        self.ckpt_path = ckpt_path

        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"找不到 linear probe 权重: {self.ckpt_path}\n"
                "请确认 checkpoints/AIGCDetectionBenchmark/linear_probe.pth 是否存在。"
            )

        print(f"[*] 使用设备: {self.device}")
        print(f"[*] 正在加载 DINOv3 Backbone: {self.model_name}")
        self.model = DinoV3Model(model_name=self.model_name, pool_type="patch_avg").to(self.device)
        self.model.eval()

        self.num_prefix = getattr(self.model.backbone, "num_prefix_tokens", 5)

        print(f"[*] 正在加载 linear probe: {self.ckpt_path}")
        ckpt = torch.load(str(self.ckpt_path), map_location=self.device)

        official_fisher_indices = ckpt.get("token_indices", None)
        if official_fisher_indices is None or len(official_fisher_indices) == 0:
            raise RuntimeError("linear_probe.pth 中缺少 token_indices，无法进行当前 System 1 推理。")

        # 转成普通 list，避免不同设备上的高级索引问题
        if isinstance(official_fisher_indices, torch.Tensor):
            official_fisher_indices = official_fisher_indices.detach().cpu().tolist()

        self.official_fisher_indices = [int(x) for x in official_fisher_indices]

        probe_state = ckpt.get("probe_state_dict", ckpt.get("model_state_dict", None))
        if probe_state is None:
            raise RuntimeError("linear_probe.pth 中缺少 probe_state_dict 或 model_state_dict。")

        in_dim = probe_state["fc.weight"].shape[1]
        out_dim = probe_state["fc.weight"].shape[0]

        self.linear_probe = LinearProbe(input_dim=in_dim, num_classes=out_dim).to(self.device)
        self.linear_probe.load_state_dict(probe_state, strict=True)
        self.linear_probe.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        print("[+] System 1 Demo Detector 初始化完成。")

    def _final_label(self, fake_prob: float) -> str:
        """
        三段式输出比硬二分类更适合公开演示。
        """
        if fake_prob >= self.fake_high_threshold:
            return "Likely AI-generated"
        if fake_prob <= self.fake_low_threshold:
            return "Likely real"
        return "Uncertain"

    def predict_pil(self, image: Image.Image) -> Tuple[Dict[str, Any], Image.Image]:
        """
        输入 PIL Image，返回:
        - result: 结构化证据 JSON
        - overlay: 原图尺寸上的可视化结果
        """
        if image is None:
            raise ValueError("输入图像为空。")

        raw_img = image.convert("RGB")
        resized_224 = raw_img.resize((224, 224), resample=Image.BICUBIC)

        input_tensor = self.transform(raw_img).unsqueeze(0).to(self.device)
        gray_img = tensor_to_gray_uint8(input_tensor)

        with torch.inference_mode():
            outputs = self.model(input_tensor, return_protocol=True)
            token_sequence = outputs["token_sequence"]

            # 1. 全局真假概率
            selected_tokens = token_sequence[:, self.official_fisher_indices, :]
            probe_logits = self.linear_probe(selected_tokens)
            probs = torch.softmax(probe_logits, dim=-1)[0]
            fake_prob = float(probs[FAKE_CLASS_IDX].detach().cpu().item())

            binary_pred_label = "Fake" if fake_prob >= 0.5 else "Real"
            final_label = self._final_label(fake_prob)

            # 2. token-level suspiciousness scores
            patch_tokens = token_sequence[:, self.num_prefix:, :]
            w_real = self.linear_probe.fc.weight[REAL_CLASS_IDX]
            w_fake = self.linear_probe.fc.weight[FAKE_CLASS_IDX]

            delta_w_norm = torch.nn.functional.normalize(w_fake - w_real, dim=0)
            patch_norm = torch.nn.functional.normalize(patch_tokens[0], dim=-1)
            forensic_scores = torch.matmul(patch_norm, delta_w_norm)

            _, top_k_indices_patch = torch.topk(
                forensic_scores,
                k=min(10, forensic_scores.shape[0]),
            )

        runtime_top_k_abs = (top_k_indices_patch + self.num_prefix).detach().cpu().tolist()
        runtime_top_k_abs = [int(x) for x in runtime_top_k_abs]

        # 3. patch 聚类
        clusters = cluster_patch_indices(runtime_top_k_abs, num_prefix=self.num_prefix)

        num_raw_clusters = len(clusters)
        num_singletons = sum(1 for c in clusters if len(c) < 2)

        # 4. 构建候选证据和验证证据
        proposal_objects, validated_evidence_objects, evidence_coverage = build_evidence_objects(
            clusters=clusters,
            forensic_scores=forensic_scores,
            gray_img=gray_img,
            num_prefix=self.num_prefix,
            fake_prob=fake_prob,
            runtime_top_k_abs=runtime_top_k_abs,
        )

        num_valid_clusters = len(validated_evidence_objects)

        pixel_any_passed = any(obj.get("pixel_any_passed", False) for obj in proposal_objects)
        token_any_passed = any(obj.get("token_any_passed", False) for obj in proposal_objects)
        has_any_passed = any(obj.get("is_validated", False) for obj in proposal_objects)
        has_high_conf = any(
            obj.get("evidence_level") == "HighConfidenceEvidence"
            for obj in validated_evidence_objects
        )
        rescued_by_token = any(obj.get("rescued_by_token", False) for obj in validated_evidence_objects)

        if validated_evidence_objects:
            best_obj = validated_evidence_objects[0]
        elif proposal_objects:
            best_obj = sorted(
                proposal_objects,
                key=lambda x: (
                    float(x.get("score_max", 0.0)),
                    float(x.get("token_contrast", 0.0)),
                    float(x.get("contrast_ratio", 0.0)),
                ),
                reverse=True,
            )[0]
        else:
            best_obj = None

        best_score = float(best_obj.get("score_max", 0.0)) if best_obj else 0.0
        best_ratio = float(best_obj.get("contrast_ratio", 0.0)) if best_obj else 0.0
        best_token_contrast = float(best_obj.get("token_contrast", 0.0)) if best_obj else 0.0

        if has_high_conf:
            status = "HighConfidenceEvidence"
        elif has_any_passed:
            status = "WeakEvidence"
        else:
            status = "NoEvidence"

        # 5. 坐标映射回原图
        proposal_objects_raw = attach_raw_geometry(proposal_objects, raw_img.size)
        validated_evidence_objects_raw = attach_raw_geometry(validated_evidence_objects, raw_img.size)

        overlay = draw_overlay(
            image=raw_img,
            validated_objects=validated_evidence_objects_raw,
            proposal_objects=proposal_objects_raw,
        )

        raw_w, raw_h = raw_img.size
        model_w, model_h = resized_224.size

        result = {
            "pred_label": final_label,
            "binary_pred_label": binary_pred_label,
            "fake_prob": fake_prob,
            "system1_status": status,

            "image_size_raw": [raw_w, raw_h],
            "image_size_model": [model_w, model_h],

            "global_thresholds": {
                "SCORE_TH": SCORE_TH,
                "RATIO_TH": RATIO_TH,
                "TOKEN_TH": TOKEN_TH,
                "GLOBAL_FAKE_GATE": GLOBAL_FAKE_GATE,
                "fake_high_threshold": self.fake_high_threshold,
                "fake_low_threshold": self.fake_low_threshold,
            },

            "num_raw_clusters": num_raw_clusters,
            "num_singletons": num_singletons,
            "num_valid_clusters": num_valid_clusters,

            "best_score_max": best_score,
            "best_contrast_ratio": best_ratio,
            "best_token_contrast": best_token_contrast,

            "pixel_any_passed": bool(pixel_any_passed),
            "token_any_passed": bool(token_any_passed),
            "has_any_passed": bool(has_any_passed),
            "has_high_conf": bool(has_high_conf),
            "rescued_by_token": bool(rescued_by_token),
            "evidence_coverage": float(evidence_coverage),

            "validated_evidence_objects": validated_evidence_objects_raw,
            "proposal_objects": proposal_objects_raw,
        }

        return to_python(result), overlay