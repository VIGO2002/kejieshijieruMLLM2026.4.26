# local_qwen3vl/smoke_test_qwen3vl_local.py

import os
import base64
import argparse
import mimetypes
import json

from local_qwen3vl.local_vlm_backend import (
    build_local_client,
    make_text_part,
    make_image_url_part,
    local_json_call,
)


def image_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def main():
    parser = argparse.ArgumentParser(description="Smoke test local Qwen3-VL with one image")
    parser.add_argument(
        "--image",
        type=str,
        default="test_img.jpg",
        help="Path to one test image"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://127.0.0.1:8000/v1"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3-VL-8B-Instruct"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    client = build_local_client(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    system_prompt = (
        "You are a strict JSON-only visual assistant. "
        "You must return valid JSON only."
    )

    user_text = """
Please inspect the image briefly.

Return exactly this JSON schema:
{
  "image_seen": true,
  "brief_description": "...",
  "visible_objects": ["..."],
  "uncertainty_note": "..."
}

Do not include markdown.
Do not include extra text.
"""

    data_url = image_to_data_url(args.image)

    user_content_parts = [
        make_image_url_part(data_url),
        make_text_part(user_text),
    ]

    parsed, raw_text, raw_response, usage = local_json_call(
        client=client,
        model_name=args.model,
        system_prompt=system_prompt,
        user_content_parts=user_content_parts,
        temperature=0.1,
        max_tokens=args.max_tokens,
    )

    print("=" * 80)
    print("[+] Parsed JSON")
    print("=" * 80)
    print(json.dumps(parsed, indent=2, ensure_ascii=False))

    print("=" * 80)
    print("[+] Raw text")
    print("=" * 80)
    print(raw_text)

    print("=" * 80)
    print("[+] Usage")
    print("=" * 80)
    print(json.dumps(usage, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()