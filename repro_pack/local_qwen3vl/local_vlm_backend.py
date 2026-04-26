# local_qwen3vl/local_vlm_backend.py
#
# 本文件用于连接本地 vLLM OpenAI-compatible API（OpenAI 兼容接口）。
# 目标：不改动你现有的 Phase A / Phase B runner，只给本地版 runner 调用。
#
# 典型 base_url:
#   http://127.0.0.1:8000/v1
#
# 典型 model:
#   Qwen3-VL-8B-Instruct

import os
import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


def build_local_client(
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "EMPTY",
    timeout: float = 3600.0,
) -> OpenAI:
    """
    构建本地 OpenAI-compatible client（OpenAI 兼容客户端）。

    参数：
    - base_url: 本地 vLLM 服务地址，默认 http://127.0.0.1:8000/v1
    - api_key: vLLM 本地服务通常不校验，填 EMPTY 即可
    - timeout: 超时时间，批量多图推理建议设长一点
    """
    return OpenAI(
        api_key=api_key or "EMPTY",
        base_url=base_url,
        timeout=timeout,
    )


def list_local_models(
    client: OpenAI,
) -> List[str]:
    """
    查询本地服务当前可用模型名。
    可用于确认 --served-model-name 是否生效。
    """
    models = client.models.list()
    return [m.id for m in models.data]


def chat_completion_local(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: int = 1024,
    top_p: float = 0.9,
    request_timeout: Optional[float] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    调用本地 Qwen3-VL。

    返回：
    - text: 模型输出文本
    - raw: OpenAI SDK 返回的完整 dict
    - usage: token 使用信息，vLLM 可能返回不完整，需兼容为空

    注意：
    - 本地 vLLM 不强依赖 response_format={"type": "json_object"}。
    - 你后续应该继续用 extract_json_from_text() 做容错解析。
    """
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    if request_timeout is not None:
        kwargs["timeout"] = request_timeout

    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    start_time = time.time()
    completion = client.chat.completions.create(**kwargs)
    elapsed = time.time() - start_time

    raw = completion.model_dump()
    text = completion.choices[0].message.content or ""
    usage = raw.get("usage", {}) or {}
    usage["_elapsed_sec"] = elapsed

    return text, raw, usage


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    从模型输出中尽量提取 JSON。

    支持：
    1. 纯 JSON
    2. ```json ... ``` fenced code block（代码块）
    3. 文本中夹杂 JSON object（JSON 对象）

    如果失败，抛出 ValueError。
    """
    if text is None:
        raise ValueError("Empty model output: None")

    s = text.strip()
    if not s:
        raise ValueError("Empty model output")

    # 1) 直接解析
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) 解析 fenced json block
    fenced = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3) 找第一个大括号到最后一个大括号
    left = s.find("{")
    right = s.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidate = s[left:right + 1].strip()
        try:
            return json.loads(candidate)
        except Exception as e:
            raise ValueError(
                f"Failed to parse extracted JSON candidate: {repr(e)}\n"
                f"Candidate prefix: {candidate[:500]}"
            )

    raise ValueError(f"No JSON object found in model output. Prefix: {s[:500]}")


def make_text_part(text: str) -> Dict[str, Any]:
    """
    OpenAI-compatible multimodal content（多模态内容）里的文本片段。
    """
    return {
        "type": "text",
        "text": text,
    }


def make_image_url_part(data_url: str) -> Dict[str, Any]:
    """
    OpenAI-compatible multimodal content（多模态内容）里的图片片段。

    data_url 示例：
    data:image/png;base64,...
    """
    return {
        "type": "image_url",
        "image_url": {
            "url": data_url,
        },
    }


def build_messages(
    system_prompt: str,
    user_content_parts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    构建标准 messages。
    user_content_parts 可以包含 text part 和 image_url part。
    """
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_content_parts,
        },
    ]


def local_json_call(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_content_parts: List[Dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: int = 1024,
    top_p: float = 0.9,
) -> Tuple[Dict[str, Any], str, Dict[str, Any], Dict[str, Any]]:
    """
    一步完成：
    1. 构建 messages
    2. 调本地模型
    3. 从文本中解析 JSON

    返回：
    - parsed_json
    - raw_text
    - raw_response
    - usage
    """
    messages = build_messages(
        system_prompt=system_prompt,
        user_content_parts=user_content_parts,
    )

    raw_text, raw_response, usage = chat_completion_local(
        client=client,
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    parsed = extract_json_from_text(raw_text)
    return parsed, raw_text, raw_response, usage


def quick_health_check(
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "EMPTY",
    model_name: str = "Qwen3-VL-8B-Instruct",
) -> None:
    """
    简单健康检查：
    - 能否连接服务
    - 能否列出模型
    - 能否完成一次纯文本调用
    """
    client = build_local_client(base_url=base_url, api_key=api_key)

    print("[*] Checking local vLLM server...")
    try:
        models = list_local_models(client)
        print(f"[+] Available models: {models}")
    except Exception as e:
        print(f"[!] Failed to list models: {repr(e)}")
        raise

    print("[*] Sending a minimal chat request...")
    messages = [
        {
            "role": "system",
            "content": "You are a JSON-only assistant.",
        },
        {
            "role": "user",
            "content": "Return exactly this JSON: {\"ok\": true}",
        },
    ]

    text, raw, usage = chat_completion_local(
        client=client,
        model_name=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=64,
    )

    print("[+] Response text:")
    print(text)
    print("[+] Usage:")
    print(json.dumps(usage, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    quick_health_check(
        base_url=os.environ.get("LOCAL_QWEN3VL_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.environ.get("LOCAL_QWEN3VL_API_KEY", "EMPTY"),
        model_name=os.environ.get("LOCAL_QWEN3VL_MODEL", "Qwen3-VL-8B-Instruct"),
    )