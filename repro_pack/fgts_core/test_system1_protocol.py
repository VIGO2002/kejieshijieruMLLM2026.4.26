import torch
import json
import os
import sys
import cv2
import numpy as np
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms
from models.dinov3_models import DinoV3Model

# ===============================================================
# 全局常量定义
# ===============================================================
REAL_CLASS_IDX = 0
FAKE_CLASS_IDX = 1

def compute_entropy(prob):
    prob = torch.clamp(prob, 1e-7, 1 - 1e-7)
    entropy = -prob * torch.log(prob) - (1 - prob) * torch.log(1 - prob)
    return entropy.item()

# ===============================================================
# 模块 A：基于网格的空间证据聚合
# ===============================================================
def absolute_idx_to_patch_rc(abs_idx, num_prefix=5, grid_size=14):
    patch_idx = abs_idx - num_prefix
    if patch_idx < 0:
        return None
    r = patch_idx // grid_size
    c = patch_idx % grid_size
    return r, c

def cluster_patch_indices(top_k_abs_indices, num_prefix=5, grid_size=14, neighbor_radius=1):
    coords = []
    for idx in top_k_abs_indices:
        rc = absolute_idx_to_patch_rc(idx, num_prefix=num_prefix, grid_size=grid_size)
        if rc is not None:
            coords.append((idx, rc[0], rc[1]))

    visited = set()
    clusters = []

    for i, (_, r, c) in enumerate(coords):
        if i in visited:
            continue
        stack = [i]
        visited.add(i)
        cluster = []

        while stack:
            cur = stack.pop()
            idx_cur, r_cur, c_cur = coords[cur]
            cluster.append((idx_cur, r_cur, c_cur))

            for j, (_, r2, c2) in enumerate(coords):
                if j in visited:
                    continue
                # Chebyshev distance
                if max(abs(r_cur - r2), abs(c_cur - c2)) <= neighbor_radius:
                    visited.add(j)
                    stack.append(j)

        clusters.append(cluster)
    return clusters

def cluster_to_bbox(cluster, patch_size=16):
    rows = [x[1] for x in cluster]
    cols = [x[2] for x in cluster]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    x1 = cmin * patch_size
    y1 = rmin * patch_size
    x2 = (cmax + 1) * patch_size
    y2 = (rmax + 1) * patch_size
    return [x1, y1, x2, y2]

# ===============================================================
# 模块 B：最小局部统计过滤器
# ===============================================================
def tensor_to_gray_uint8(img_tensor):
    img_np = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return gray

def validate_region_with_neighbor(gray_img, bbox, expand=16, threshold_ratio=1.25):
    H, W = gray_img.shape
    x1, y1, x2, y2 = bbox
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    
    ex1, ey1 = max(0, x1 - expand), max(0, y1 - expand)
    ex2, ey2 = min(W, x2 + expand), min(H, y2 + expand)

    expanded_gray = gray_img[ey1:ey2, ex1:ex2]
    lap_map = cv2.Laplacian(expanded_gray, cv2.CV_64F)

    rx1, ry1 = x1 - ex1, y1 - ey1
    rx2, ry2 = x2 - ex1, y2 - ey1
    
    region_mask = np.zeros(expanded_gray.shape, dtype=bool)
    region_mask[ry1:ry2, rx1:rx2] = True
    
    ring_mask = np.ones(expanded_gray.shape, dtype=bool)
    ring_mask[ry1:ry2, rx1:rx2] = False 

    region_lap_vals = lap_map[region_mask]
    ring_lap_vals = lap_map[ring_mask]

    if region_lap_vals.size == 0 or ring_lap_vals.size < 25:
        # 【修改3】：返回四个值，保留方差供后续统计
        return False, 0.0, 0.0, 0.0

    region_var = region_lap_vals.var()
    neighbor_var = ring_lap_vals.var()
    
    ratio = region_var / (neighbor_var + 1e-6)
    is_valid = ratio > threshold_ratio
    
    # 【修改3】：返回四个值
    return is_valid, round(ratio, 3), round(region_var, 3), round(neighbor_var, 3)

# ===============================================================
# 主模型结构
# ===============================================================
class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.fc(x)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 当前使用设备: {device}")
    
    print("[*] 正在加载 DINOv3 Backbone...")
    model = DinoV3Model(model_name='dinov3_vit_7b', pool_type='patch_avg').to(device)
    model.eval()
    num_prefix = getattr(model.backbone, "num_prefix_tokens", 5)
    
    weight_path = 'checkpoints/AIGCDetectionBenchmark/linear_probe.pth'
    official_fisher_indices = None
    probe_loaded = False
    linear_probe = None

    if os.path.exists(weight_path):
        ckpt = torch.load(weight_path, map_location=device)
        official_fisher_indices = ckpt.get("token_indices", None)
        probe_state = ckpt.get("probe_state_dict", ckpt.get("model_state_dict", None))
        
        if probe_state is not None:
            in_dim = probe_state["fc.weight"].shape[1]
            out_dim = probe_state["fc.weight"].shape[0]
            linear_probe = LinearProbe(input_dim=in_dim, num_classes=out_dim).to(device)
            linear_probe.eval()
            linear_probe.load_state_dict(probe_state, strict=True)
            probe_loaded = True
            
    # 【修改1】：更严格的保护逻辑，包含官方索引的检查
    if not probe_loaded or linear_probe is None or official_fisher_indices is None:
        raise RuntimeError("[!] 严重错误: 线性探针或 Fisher 索引加载失败。提取法医分数必须依赖该权重！")
            
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_img.jpg"
    if not os.path.exists(image_path):
        print(f"[!] 找不到真实的测试图片: {image_path}")
        return
        
    print(f"[*] 正在加载测试图像: {image_path}...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    raw_img = Image.open(image_path).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0).to(device)
    gray_img = tensor_to_gray_uint8(input_tensor)
    
    print("[*] 正在执行前向传播...")
    with torch.no_grad():
        outputs = model(input_tensor, return_protocol=True)
    token_sequence = outputs["token_sequence"] 
    
    # 【修改1】：删除 fallback，直接使用严格路径
    selected_tokens = token_sequence[:, official_fisher_indices, :]
    probe_logits = linear_probe(selected_tokens)
    probs = torch.softmax(probe_logits, dim=-1)[0]
    fake_prob = probs[FAKE_CLASS_IDX].item() 
        
    entropy = compute_entropy(torch.tensor(fake_prob))
    
    # --- 1. 提取法医异常打分 ---
    patch_tokens = token_sequence[:, num_prefix:, :] 
    w_real = linear_probe.fc.weight[REAL_CLASS_IDX] 
    w_fake = linear_probe.fc.weight[FAKE_CLASS_IDX] 
    delta_w = w_fake - w_real 
    
    delta_w_norm = torch.nn.functional.normalize(delta_w, dim=0)
    patch_norm = torch.nn.functional.normalize(patch_tokens[0], dim=-1)
    forensic_scores = torch.matmul(patch_norm, delta_w_norm) 
    
    num_initial_patches = 10
    top_k_values, top_k_indices_patch = torch.topk(forensic_scores, k=min(num_initial_patches, forensic_scores.shape[0]))
    runtime_top_k_absolute = (top_k_indices_patch + num_prefix).cpu().tolist()
    
    # --- 2. 空间聚合 ---
    clusters = cluster_patch_indices(runtime_top_k_absolute, num_prefix=num_prefix)
    num_clusters = len(clusters)
    
    # --- 3. 最小局部统计过滤 ---
    validated_evidence = []
    num_passed_regions = 0
    
    for cluster in clusters:
        bbox = cluster_to_bbox(cluster)
        source_indices = [x[0] for x in cluster]
        
        cluster_scores = [forensic_scores[idx - num_prefix].item() for idx in source_indices]
        score_mean = sum(cluster_scores) / len(cluster_scores)
        score_max = max(cluster_scores)
        
        # 【修改3】：接收并记录新增的方差变量
        is_valid, contrast_ratio, reg_var, neigh_var = validate_region_with_neighbor(gray_img, bbox)
        
        # 【修改2】：状态命中性化
        status = "Passed" if is_valid else "RejectedByLocalFilter"
        if is_valid: 
            num_passed_regions += 1
            
        validated_evidence.append({
            "region_bbox": bbox,
            "source_patch_indices": source_indices,
            "aggregation_score_mean": round(score_mean, 4),
            "aggregation_score_max": round(score_max, 4),
            "validation_metric": "laplacian_neighbor_ratio",
            "statistical_contrast_score": contrast_ratio,
            "region_laplacian_variance": reg_var,      # 新增
            "neighbor_laplacian_variance": neigh_var,  # 新增
            "validation_status": status
        })
        
    # 【修改4】：双重条件排序 (先 Passed，再按得分降序)
    validated_evidence.sort(
        key=lambda x: (x["validation_status"] == "Passed", x["aggregation_score_max"]), 
        reverse=True
    )
            
    system_1_output = {
        "image_id": image_path,
        "classification": {
            "fake_probability": round(fake_prob, 4),
            "entropy": round(entropy, 4)
        },
        "discriminative_support": {
            "official_fisher_token_indices": official_fisher_indices if official_fisher_indices else []
        },
        "evidence_summary": {
            "num_initial_patches": num_initial_patches,
            "num_clusters": num_clusters,
            "num_passed_regions": num_passed_regions,
            "description": f"Extracted {num_initial_patches} raw patches, clustered into {num_clusters} spatial regions. {num_passed_regions} passed the local statistical filter."
        },
        "validated_evidence_regions": validated_evidence
    }
    
    print("\n========== System 1 Output Protocol ==========")
    print(json.dumps(system_1_output, indent=4, ensure_ascii=False))
    print("==============================================\n")

    # --- 4. 生成可视化叠加图 ---
    print("[*] 正在生成过滤后的视觉证据图...")
    original_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original_img.size
    scale_x, scale_y = orig_w / 224.0, orig_h / 224.0
    draw = ImageDraw.Draw(original_img)
    
    for ev in validated_evidence:
        if ev["validation_status"] == "Passed":
            x1, y1, x2, y2 = ev["region_bbox"]
            orig_x1, orig_y1 = x1 * scale_x, y1 * scale_y
            orig_x2, orig_y2 = x2 * scale_x, y2 * scale_y
            draw.rectangle([orig_x1, orig_y1, orig_x2, orig_y2], outline="red", width=4)
    
    overlay_save_path = f"validated_{os.path.basename(image_path)}"
    original_img.save(overlay_save_path)
    print(f"[+] 验证后视觉证据图已保存至: {overlay_save_path}\n")
    
if __name__ == "__main__":
    main()