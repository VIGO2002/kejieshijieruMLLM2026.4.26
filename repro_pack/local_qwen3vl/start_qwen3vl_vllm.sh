#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Start local Qwen3-VL-8B-Instruct with vLLM
# ============================================================
#
# 用法：
#   bash local_qwen3vl/start_qwen3vl_vllm.sh
#
# 可选环境变量：
#   MODEL_DIR=/root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct
#   SERVED_MODEL_NAME=Qwen3-VL-8B-Instruct
#   HOST=0.0.0.0
#   PORT=8000
#   CUDA_VISIBLE_DEVICES=0
#   MAX_MODEL_LEN=32768
#   GPU_MEMORY_UTILIZATION=0.88
#
# 启动后接口：
#   http://127.0.0.1:8000/v1
#
# 后续 Python client 里的 model_name 建议使用：
#   Qwen3-VL-8B-Instruct

# 你可以按实际下载位置修改这里。
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-8B-Instruct}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 31GB 显存建议先用 32768。后面如果显存富余再加大。
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# 多图输入 + 长 prompt 会吃 KV cache（键值缓存），先不要拉满。
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

# 每个请求最多允许多少张图片。你的 Phase A 通常 raw + overlay + crops，设 8 比较稳。
LIMIT_IMAGES_PER_PROMPT="${LIMIT_IMAGES_PER_PROMPT:-8}"

export CUDA_VISIBLE_DEVICES

echo "============================================================"
echo "[*] Starting Qwen3-VL local vLLM server"
echo "------------------------------------------------------------"
echo "MODEL_DIR              = ${MODEL_DIR}"
echo "SERVED_MODEL_NAME      = ${SERVED_MODEL_NAME}"
echo "HOST                   = ${HOST}"
echo "PORT                   = ${PORT}"
echo "CUDA_VISIBLE_DEVICES   = ${CUDA_VISIBLE_DEVICES}"
echo "MAX_MODEL_LEN          = ${MAX_MODEL_LEN}"
echo "GPU_MEMORY_UTILIZATION = ${GPU_MEMORY_UTILIZATION}"
echo "LIMIT_IMAGES_PER_PROMPT= ${LIMIT_IMAGES_PER_PROMPT}"
echo "============================================================"

if [ ! -d "${MODEL_DIR}" ]; then
  echo "[!] MODEL_DIR does not exist: ${MODEL_DIR}"
  echo "    请确认 ModelScope 下载后的真实路径。"
  exit 1
fi

# 建议先确认 vllm 是否可用
python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("vllm")
if spec is None:
    raise SystemExit("[!] vllm is not installed. Please install vllm first.")
print("[+] vllm is installed.")
PY

# 启动 OpenAI-compatible API server（OpenAI 兼容服务）
#
# 说明：
# --served-model-name:
#   让客户端使用一个固定短名字，而不是本地长路径。
#
# --trust-remote-code:
#   Qwen 系列多模态模型通常建议开启。
#
# --limit-mm-per-prompt:
#   限制每个 prompt 的多模态输入数量，避免误传太多图片。
#
# 如果你的 vLLM 版本不支持 --limit-mm-per-prompt，请删除这一行参数。
LIMIT_MM_PER_PROMPT_JSON="{\"image\": ${LIMIT_IMAGES_PER_PROMPT}}"

vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT_JSON}"