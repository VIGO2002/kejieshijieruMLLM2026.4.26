import os
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

# 直接复用当前冻结的 System 1 规则与辅助函数
from cross_domain_eval import (
    SCORE_TH,
    RATIO_TH,
    TOKEN_TH,
    GLOBAL_FAKE_GATE,
    build_evidence_objects,
)
from test_system1_protocol import (
    DinoV3Model,
    LinearProbe,
    REAL_CLASS_IDX,
    FAKE_CLASS_IDX,
    tensor_to_gray_uint8,
    cluster_patch_indices,
)

# ===============================================================
# 18 张样本（9 fake + 9 real）
# ===============================================================
SELECTED_CASES = [
    # ---------------------------
    # Fake / HighConfidenceEvidence
    # ---------------------------
    {
        "case_tag": "Fake_High",
        "filename": "000017.png",
        "dataset_name": "Group 1: GANs",
        "image_path": "batch_eval_total/batch_eval_gans/fake/000017.png",
        "label": "Fake",
    },
    {
        "case_tag": "Fake_High",
        "filename": "007_sdv5_00027.png",
        "dataset_name": "Group 2: Diffusion",
        "image_path": "batch_eval_total/batch_eval_diffusion/fake/007_sdv5_00027.png",
        "label": "Fake",
    },
    {
        "case_tag": "Fake_High",
        "filename": "aaksjqrtil.png",
        "dataset_name": "Group 3a: Guided",
        "image_path": "batch_eval_total/batch_eval_OOD/fake/aaksjqrtil.png",
        "label": "Fake",
    },

    # ---------------------------
    # Fake / WeakEvidence
    # ---------------------------
    {
        "case_tag": "Fake_Weak",
        "filename": "007_sdv5_00039.png",
        "dataset_name": "Group 2: Diffusion",
        "image_path": "batch_eval_total/batch_eval_diffusion/fake/007_sdv5_00039.png",
        "label": "Fake",
    },
    {
        "case_tag": "Fake_Weak",
        "filename": "bsoomuiluf.png",
        "dataset_name": "Group 3a: Guided",
        "image_path": "batch_eval_total/batch_eval_OOD/fake/bsoomuiluf.png",
        "label": "Fake",
    },
    {
        "case_tag": "Fake_Weak",
        "filename": "127_midjourney_197.png",
        "dataset_name": "Group 3b: Midjourney",
        "image_path": "batch_eval_total/batch_eval_OOD1/fake/127_midjourney_197.png",
        "label": "Fake",
    },

    # ---------------------------
    # Fake / NoEvidence
    # ---------------------------
    {
        "case_tag": "Fake_None",
        "filename": "058_sdv5_00003.png",
        "dataset_name": "Group 2: Diffusion",
        "image_path": "batch_eval_total/batch_eval_diffusion/fake/058_sdv5_00003.png",
        "label": "Fake",
    },
    {
        "case_tag": "Fake_None",
        "filename": "bbmeviacjm.png",
        "dataset_name": "Group 3a: Guided",
        "image_path": "batch_eval_total/batch_eval_OOD/fake/bbmeviacjm.png",
        "label": "Fake",
    },
    {
        "case_tag": "Fake_None",
        "filename": "150_midjourney_100.png",
        "dataset_name": "Group 3b: Midjourney",
        "image_path": "batch_eval_total/batch_eval_OOD1/fake/150_midjourney_100.png",
        "label": "Fake",
    },

    # ---------------------------
    # Real
    # ---------------------------
    {
        "case_tag": "Real",
        "filename": "00013.png",
        "dataset_name": "Group 1: GANs",
        "image_path": "batch_eval_total/batch_eval_gans/real/00013.png",
        "label": "Real",
    },
    {
        "case_tag": "Real",
        "filename": "00015.png",
        "dataset_name": "Group 1: GANs",
        "image_path": "batch_eval_total/batch_eval_gans/real/00015.png",
        "label": "Real",
    },
    {
        "case_tag": "Real",
        "filename": "00024.png",
        "dataset_name": "Group 1: GANs",
        "image_path": "batch_eval_total/batch_eval_gans/real/00024.png",
        "label": "Real",
    },
    {
        "case_tag": "Real",
        "filename": "ILSVRC2012_val_00000098.JPEG",
        "dataset_name": "Group 2: Diffusion",
        "image_path": "batch_eval_total/batch_eval_diffusion/real/ILSVRC2012_val_00000098.JPEG",
        "label": "Real",
    },
    {
        "case_tag": "Real",
        "filename": "ILSVRC2012_val_00000191.JPEG",
        "dataset_name": "Group 2: Diffusion",
        "image_path": "batch_eval_total/batch_eval_diffusion/real/ILSVRC2012_val_00000191.JPEG",
        "label": "Real",
    },
    {
        "case_tag": "Real",
        "filename": "ILSVRC2012_val_00000328.JPEG",
        "dataset_name": "Group 2: Diffusion",
        "image_path": "batch_eval_total/batch_eval_diffusion/real/ILSVRC2012_val_00000328.JPEG",
        "label": "Real",
    },
    {
        "case_tag": "Real",
        "filename": "cmjapngcpn.JPEG",
        "dataset_name": "Group 3a: Guided",
        "image_path": "batch_eval_total/batch_eval_OOD/real/cmjapngcpn.JPEG",
        "label": "Real",
    },
    {
        "case_tag": "Real",
        "filename": "cxjsjqusph.JPEG",
        "dataset_name": "Group 3a: Guided",
        "image_path": "batch_eval_total/batch_eval_OOD/real/cxjsjqusph.JPEG",
        "label": "Real",
    },
    {
        "case_tag": "Real",
        "filename": "ILSVRC2012_val_00000446.JPEG",
        "dataset_name": "Group 3b: Midjourney",
        "image_path": "batch_eval_total/batch_eval_OOD1/real/ILSVRC2012_val_00000446.JPEG",
        "label": "Real",
    },
]

OUTPUT_ROOT = "system2_case_exports_18"
CROP_PADDING_RATIO = 0.15  # crop 外扩比例


# ===============================================================
# 工具函数
# ===============================================================
def to_python(obj: Any) -> Any:
    """把 tensor / numpy 类型递归转成 JSON 可序列化的 Python 基本类型。"""
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()

    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_python(x) for x in obj]

    if isinstance(obj, tuple):
        return [to_python(x) for x in obj]

    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    return obj


def get_source_type(dataset_name: str, label: str) -> str:
    if dataset_name == "Group 1: GANs":
        return "StyleGAN2" if label == "Fake" else "Real_Cars"
    elif dataset_name == "Group 2: Diffusion":
        return "SDv1.5" if label == "Fake" else "ImageNet_Real"
    elif dataset_name == "Group 3a: Guided":
        return "Guided" if label == "Fake" else "ImageNet_Real"
    elif dataset_name == "Group 3b: Midjourney":
        return "Midjourney" if label == "Fake" else "ImageNet_Real"
    return "Unknown"


def scale_bbox_224_to_raw(bbox_224: List[int], raw_size: Tuple[int, int]) -> List[int]:
    """把 224 坐标映射回原图尺寸。"""
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


def expand_bbox(bbox: List[int], img_size: Tuple[int, int], padding_ratio: float = 0.15) -> List[int]:
    """对 bbox 做一定外扩，用于 crop。"""
    img_w, img_h = img_size
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(round(bw * padding_ratio))
    pad_y = int(round(bh * padding_ratio))

    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(img_w, x2 + pad_x)
    ny2 = min(img_h, y2 + pad_y)

    return [nx1, ny1, nx2, ny2]


def draw_overlay(image: Image.Image, objects: List[Dict[str, Any]], proposals: List[Dict[str, Any]], mode: str = "224") -> Image.Image:
    """
    mode:
    - "224": 使用 bbox_224
    - "raw": 使用 bbox_raw
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    bbox_key = "bbox_224" if mode == "224" else "bbox_raw"

    if len(objects) > 0:
        for obj in objects:
            bbox = obj[bbox_key]
            level = obj["evidence_level"]
            rank = obj.get("rank", -1)

            if level == "HighConfidenceEvidence":
                color = (255, 0, 0)      # red
                width = 3
            else:
                color = (255, 215, 0)    # yellow
                width = 3

            draw.rectangle(bbox, outline=color, width=width)

            x1, y1, x2, y2 = bbox
            tag_w, tag_h = 22, 16
            draw.rectangle([x1, max(0, y1 - tag_h), x1 + tag_w, y1], fill=color)
            draw.text((x1 + 4, max(0, y1 - tag_h + 1)), str(rank), fill=(0, 0, 0))
    else:
        # NoEvidence 时画 proposal top-1 作为参考框（蓝色）
        if len(proposals) > 0:
            top_obj = sorted(
                proposals,
                key=lambda obj: (obj["score_max"], obj["token_contrast"], obj["contrast_ratio"]),
                reverse=True
            )[0]
            bbox = top_obj[bbox_key]
            draw.rectangle(bbox, outline=(30, 144, 255), width=2)

    return overlay


def attach_raw_geometry(objects: List[Dict[str, Any]], raw_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    """给 proposal / validated evidence object 加上 bbox_raw。"""
    new_objects = []
    for obj in objects:
        new_obj = dict(obj)
        new_obj["bbox_raw"] = scale_bbox_224_to_raw(obj["bbox_224"], raw_size)
        new_objects.append(new_obj)
    return new_objects


def sanitize_for_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    给大模型看的 JSON：
    - 删除 label / pred_label / cls_correct / source_type / dataset 等可能泄漏答案的字段
    - 只保留法医解释真正需要的内容
    """
    keep_keys = [
        "filename",
        "image_size_raw",
        "image_size_model",
        "global_thresholds",
        "fake_prob",
        "status",
        "evidence_coverage",
        "num_raw_clusters",
        "num_singletons",
        "num_valid_clusters",
        "best_score_max",
        "best_contrast_ratio",
        "best_token_contrast",
        "pixel_any_passed",
        "token_any_passed",
        "has_any_passed",
        "has_high_conf",
        "rescued_by_token",
        "validated_evidence_objects",
        "proposal_objects",
    ]
    return {k: payload[k] for k in keep_keys if k in payload}


# ===============================================================
# System 1 单图推理
# ===============================================================
def load_system1(device: str):
    model = DinoV3Model(model_name='dinov3_vit_7b', pool_type='patch_avg').to(device)
    model.eval()
    num_prefix = getattr(model.backbone, "num_prefix_tokens", 5)

    ckpt = torch.load('checkpoints/AIGCDetectionBenchmark/linear_probe.pth', map_location=device)
    official_fisher_indices = ckpt.get("token_indices", None)
    assert official_fisher_indices is not None and len(official_fisher_indices) > 0, "[!] 致命错误: token_indices 缺失！"

    probe_state = ckpt.get("probe_state_dict", ckpt.get("model_state_dict", None))
    assert probe_state is not None, "[!] 致命错误: probe_state_dict 缺失！"

    in_dim, out_dim = probe_state["fc.weight"].shape[1], probe_state["fc.weight"].shape[0]
    linear_probe = LinearProbe(input_dim=in_dim, num_classes=out_dim).to(device)
    linear_probe.load_state_dict(probe_state, strict=True)
    linear_probe.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    return model, linear_probe, official_fisher_indices, num_prefix, transform


def evaluate_single_image(
    image_path: str,
    dataset_name: str,
    label: str,
    model,
    linear_probe,
    official_fisher_indices,
    num_prefix,
    device: str,
    transform
) -> Dict[str, Any]:
    raw_img = Image.open(image_path).convert('RGB')
    resized_224 = raw_img.resize((224, 224), resample=Image.BICUBIC)

    input_tensor = transform(raw_img).unsqueeze(0).to(device)
    gray_img = tensor_to_gray_uint8(input_tensor)

    with torch.no_grad():
        outputs = model(input_tensor, return_protocol=True)
        token_sequence = outputs["token_sequence"]

        # 分类概率
        selected_tokens = token_sequence[:, official_fisher_indices, :]
        probe_logits = linear_probe(selected_tokens)
        probs = torch.softmax(probe_logits, dim=-1)[0]
        fake_prob = probs[FAKE_CLASS_IDX].item()

        pred_label = "Fake" if fake_prob >= 0.5 else "Real"
        cls_correct = int(pred_label == label)

        # token-level suspiciousness scores
        patch_tokens = token_sequence[:, num_prefix:, :]
        w_real = linear_probe.fc.weight[REAL_CLASS_IDX]
        w_fake = linear_probe.fc.weight[FAKE_CLASS_IDX]
        delta_w_norm = torch.nn.functional.normalize(w_fake - w_real, dim=0)
        patch_norm = torch.nn.functional.normalize(patch_tokens[0], dim=-1)
        forensic_scores = torch.matmul(patch_norm, delta_w_norm)

        _, top_k_indices_patch = torch.topk(forensic_scores, k=min(10, forensic_scores.shape[0]))

    runtime_top_k_abs = (top_k_indices_patch + num_prefix).cpu().tolist()
    clusters = cluster_patch_indices(runtime_top_k_abs, num_prefix=num_prefix)

    num_raw_clusters = len(clusters)
    num_singletons = sum(1 for c in clusters if len(c) < 2)

    proposal_objects, validated_evidence_objects, evidence_coverage = build_evidence_objects(
        clusters=clusters,
        forensic_scores=forensic_scores,
        gray_img=gray_img,
        num_prefix=num_prefix,
        fake_prob=fake_prob,
        runtime_top_k_abs=runtime_top_k_abs
    )

    num_valid_clusters = len(validated_evidence_objects)

    pixel_any_passed = any(obj["pixel_any_passed"] for obj in proposal_objects)
    token_any_passed = any(obj["token_any_passed"] for obj in proposal_objects)
    has_any_passed = any(obj["is_validated"] for obj in proposal_objects)
    has_high_conf = any(obj["evidence_level"] == "HighConfidenceEvidence" for obj in validated_evidence_objects)
    rescued_by_token = any(obj["rescued_by_token"] for obj in validated_evidence_objects)

    if validated_evidence_objects:
        best_obj = validated_evidence_objects[0]
    elif proposal_objects:
        best_obj = sorted(
            proposal_objects,
            key=lambda obj: (obj["score_max"], obj["token_contrast"], obj["contrast_ratio"]),
            reverse=True
        )[0]
    else:
        best_obj = None

    best_score = best_obj["score_max"] if best_obj else 0.0
    best_ratio = best_obj["contrast_ratio"] if best_obj else 0.0
    best_token_contrast = best_obj["token_contrast"] if best_obj else 0.0

    if has_high_conf:
        status = "HighConfidenceEvidence"
    elif has_any_passed:
        status = "WeakEvidence"
    else:
        status = "NoEvidence"

    result = {
        "dataset": dataset_name,
        "source_type": get_source_type(dataset_name, label),
        "filename": os.path.basename(image_path),
        "image_path": image_path,
        "label": label,
        "pred_label": pred_label,
        "cls_correct": cls_correct,
        "fake_prob": fake_prob,

        "global_thresholds": {
            "SCORE_TH": SCORE_TH,
            "RATIO_TH": RATIO_TH,
            "TOKEN_TH": TOKEN_TH,
            "GLOBAL_FAKE_GATE": GLOBAL_FAKE_GATE
        },

        "num_raw_clusters": num_raw_clusters,
        "num_singletons": num_singletons,
        "num_valid_clusters": num_valid_clusters,

        "best_score_max": best_score,
        "best_contrast_ratio": best_ratio,
        "best_token_contrast": best_token_contrast,

        "pixel_any_passed": pixel_any_passed,
        "token_any_passed": token_any_passed,
        "has_any_passed": has_any_passed,
        "has_high_conf": has_high_conf,
        "rescued_by_token": rescued_by_token,

        "evidence_coverage": evidence_coverage,
        "proposal_objects": proposal_objects,
        "validated_evidence_objects": validated_evidence_objects,
        "status": status
    }

    return {
        "result": result,
        "raw_img": raw_img,
        "resized_224": resized_224
    }


# ===============================================================
# 导出
# ===============================================================
def export_case(case_cfg: Dict[str, Any], export_root: str, system1_bundle, device: str):
    model, linear_probe, official_fisher_indices, num_prefix, transform = system1_bundle

    image_path = case_cfg["image_path"]
    filename = case_cfg["filename"]
    case_tag = case_cfg["case_tag"]
    dataset_name = case_cfg["dataset_name"]
    label = case_cfg["label"]

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[!] 找不到图片: {image_path}")

    pack = evaluate_single_image(
        image_path=image_path,
        dataset_name=dataset_name,
        label=label,
        model=model,
        linear_probe=linear_probe,
        official_fisher_indices=official_fisher_indices,
        num_prefix=num_prefix,
        device=device,
        transform=transform
    )

    result = pack["result"]
    raw_img = pack["raw_img"]
    resized_224 = pack["resized_224"]

    raw_w, raw_h = raw_img.size
    model_w, model_h = resized_224.size

    proposal_objects_raw = attach_raw_geometry(result["proposal_objects"], raw_img.size)
    validated_evidence_objects_raw = attach_raw_geometry(result["validated_evidence_objects"], raw_img.size)

    stem = Path(filename).stem
    out_dir = os.path.join(export_root, f"{case_tag}__{stem}")
    os.makedirs(out_dir, exist_ok=True)

    crop_dir = os.path.join(out_dir, "crops_raw")
    os.makedirs(crop_dir, exist_ok=True)

    # 1) 原始图（原尺寸）
    raw_ext = Path(filename).suffix if Path(filename).suffix else ".png"
    raw_out_path = os.path.join(out_dir, f"{stem}_original_raw{raw_ext}")
    shutil.copy(image_path, raw_out_path)

    # 2) 原图 224 版
    original_224_path = os.path.join(out_dir, f"{stem}_original_224.png")
    resized_224.save(original_224_path)

    # 3) overlay_224
    overlay_224_img = draw_overlay(
        image=resized_224,
        objects=result["validated_evidence_objects"],
        proposals=result["proposal_objects"],
        mode="224"
    )
    overlay_224_path = os.path.join(out_dir, f"{stem}_overlay_224.png")
    overlay_224_img.save(overlay_224_path)

    # 4) overlay_raw
    overlay_raw_img = draw_overlay(
        image=raw_img,
        objects=validated_evidence_objects_raw,
        proposals=proposal_objects_raw,
        mode="raw"
    )
    overlay_raw_path = os.path.join(out_dir, f"{stem}_overlay_raw.png")
    overlay_raw_img.save(overlay_raw_path)

    # 5) 导出 raw crop
    if len(validated_evidence_objects_raw) > 0:
        crop_source = validated_evidence_objects_raw
    else:
        crop_source = []
        if len(proposal_objects_raw) > 0:
            top_obj = sorted(
                proposal_objects_raw,
                key=lambda obj: (obj["score_max"], obj["token_contrast"], obj["contrast_ratio"]),
                reverse=True
            )[0]
            crop_source = [top_obj]

    for idx, obj in enumerate(crop_source, start=1):
        bbox_raw = obj["bbox_raw"]
        crop_bbox = expand_bbox(bbox_raw, raw_img.size, CROP_PADDING_RATIO)
        crop_img = raw_img.crop(crop_bbox)

        rank = obj.get("rank", idx)
        level = obj.get("evidence_level", "Proposal")
        crop_name = f"{stem}_crop_{idx:02d}_rank{rank}_{level}.png"
        crop_img.save(os.path.join(crop_dir, crop_name))

    # 6) 准备 JSON
    evidence_payload_full = {
        "case_tag": case_tag,
        "dataset": result["dataset"],
        "source_type": result["source_type"],
        "filename": result["filename"],
        "image_size_raw": [raw_w, raw_h],
        "image_size_model": [model_w, model_h],

        "label": result["label"],              # full 版保留
        "pred_label": result["pred_label"],    # full 版保留
        "cls_correct": result["cls_correct"],  # full 版保留

        "fake_prob": result["fake_prob"],
        "status": result["status"],
        "evidence_coverage": result["evidence_coverage"],
        "num_raw_clusters": result["num_raw_clusters"],
        "num_singletons": result["num_singletons"],
        "num_valid_clusters": result["num_valid_clusters"],

        "best_score_max": result["best_score_max"],
        "best_contrast_ratio": result["best_contrast_ratio"],
        "best_token_contrast": result["best_token_contrast"],

        "pixel_any_passed": result["pixel_any_passed"],
        "token_any_passed": result["token_any_passed"],
        "has_any_passed": result["has_any_passed"],
        "has_high_conf": result["has_high_conf"],
        "rescued_by_token": result["rescued_by_token"],

        "global_thresholds": result["global_thresholds"],
        "validated_evidence_objects": validated_evidence_objects_raw,
        "proposal_objects": proposal_objects_raw,
    }

    evidence_payload_model = sanitize_for_model(evidence_payload_full)

    full_json_path = os.path.join(out_dir, f"{stem}_evidence_full.json")
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(to_python(evidence_payload_full), f, indent=2, ensure_ascii=False)

    model_json_path = os.path.join(out_dir, f"{stem}_evidence_for_model.json")
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(to_python(evidence_payload_model), f, indent=2, ensure_ascii=False)

    print(f"[+] 已导出: {out_dir}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 正在初始化 System 1 导出环境 ({device})...")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    system1_bundle = load_system1(device)

    for case_cfg in SELECTED_CASES:
        export_case(
            case_cfg=case_cfg,
            export_root=OUTPUT_ROOT,
            system1_bundle=system1_bundle,
            device=device
        )

    print(f"\n[+] 所有 {len(SELECTED_CASES)} 张样本已导出到: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()