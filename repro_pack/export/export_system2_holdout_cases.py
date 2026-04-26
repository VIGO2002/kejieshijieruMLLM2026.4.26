import os
from pathlib import Path
from typing import Dict, List

# 直接复用你已经写好的导出函数
from export_system2_cases import load_system1, export_case
import torch

# ===============================================================
# 配置
# ===============================================================
LIST_FILE = "新建文本文档.txt"
OUTPUT_ROOT = "system2_holdout_exports"

GROUP_META = {
    "G1 real": {
        "dataset_name": "Group 1: GANs",
        "base_dir": "batch_eval_total/batch_eval_gans/real",
        "label": "Real",
        "group_code": "G1"
    },
    "G1 fake": {
        "dataset_name": "Group 1: GANs",
        "base_dir": "batch_eval_total/batch_eval_gans/fake",
        "label": "Fake",
        "group_code": "G1"
    },
    "G2 real": {
        "dataset_name": "Group 2: Diffusion",
        "base_dir": "batch_eval_total/batch_eval_diffusion/real",
        "label": "Real",
        "group_code": "G2"
    },
    "G2 fake": {
        "dataset_name": "Group 2: Diffusion",
        "base_dir": "batch_eval_total/batch_eval_diffusion/fake",
        "label": "Fake",
        "group_code": "G2"
    },
    "G3a real": {
        "dataset_name": "Group 3a: Guided",
        "base_dir": "batch_eval_total/batch_eval_OOD/real",
        "label": "Real",
        "group_code": "G3a"
    },
    "G3a fake": {
        "dataset_name": "Group 3a: Guided",
        "base_dir": "batch_eval_total/batch_eval_OOD/fake",
        "label": "Fake",
        "group_code": "G3a"
    },
    "G3b real": {
        "dataset_name": "Group 3b: Midjourney",
        "base_dir": "batch_eval_total/batch_eval_OOD1/real",
        "label": "Real",
        "group_code": "G3b"
    },
    "G3b fake": {
        "dataset_name": "Group 3b: Midjourney",
        "base_dir": "batch_eval_total/batch_eval_OOD1/fake",
        "label": "Fake",
        "group_code": "G3b"
    },
}


def normalize_group_header(line: str) -> str:
    line = line.strip()
    line = line.replace("：", ":")
    if line.endswith(":"):
        line = line[:-1].strip()
    return line


def parse_grouped_list(txt_path: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    current_group = None

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # 组头
            if line.endswith("：") or line.endswith(":"):
                header = normalize_group_header(line)
                if header not in GROUP_META:
                    raise ValueError(f"[!] 未知分组头: {header}")
                current_group = header
                groups[current_group] = []
                continue

            if current_group is None:
                raise ValueError(f"[!] 在出现分组头之前就遇到文件名: {line}")

            groups[current_group].append(line)

    return groups


def build_case_cfgs(groups: Dict[str, List[str]]) -> List[Dict]:
    case_cfgs = []
    for group_name, filenames in groups.items():
        meta = GROUP_META[group_name]
        for fn in filenames:
            image_path = os.path.join(meta["base_dir"], fn)
            stem = Path(fn).stem
            case_tag = f"Holdout__{meta['group_code']}__{meta['label']}"
            case_cfgs.append({
                "case_tag": case_tag,
                "filename": fn,
                "dataset_name": meta["dataset_name"],
                "image_path": image_path,
                "label": meta["label"],
            })
    return case_cfgs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 正在初始化 System 1 hold-out 导出环境 ({device})...")

    groups = parse_grouped_list(LIST_FILE)
    case_cfgs = build_case_cfgs(groups)

    num_real = sum(1 for x in case_cfgs if x["label"] == "Real")
    num_fake = sum(1 for x in case_cfgs if x["label"] == "Fake")

    print(f"[*] 从 {LIST_FILE} 解析到 {len(case_cfgs)} 个样本")
    print(f"    Real: {num_real}")
    print(f"    Fake: {num_fake}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    system1_bundle = load_system1(device)

    missing = []
    for cfg in case_cfgs:
        if not os.path.exists(cfg["image_path"]):
            missing.append(cfg["image_path"])

    if missing:
        print("[!] 以下文件不存在，请先检查路径：")
        for p in missing:
            print("   ", p)
        raise FileNotFoundError("[!] 有文件缺失，终止导出。")

    for cfg in case_cfgs:
        export_case(
            case_cfg=cfg,
            export_root=OUTPUT_ROOT,
            system1_bundle=system1_bundle,
            device=device
        )

    print(f"\n[+] hold-out 样本已全部导出到: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()