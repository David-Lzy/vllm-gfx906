#!/usr/bin/env python3
"""Run the routine text, image, and JSON gates for the Qwen3.6 TP4 A/B."""

import argparse
import base64
import io
import json
import sys
import urllib.request

from PIL import Image, ImageDraw


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def chat(base_url: str, model: str, content: list[dict], json_mode: bool) -> dict:
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def image_data_uri(color: str, label: str) -> str:
    image = Image.new("RGB", (256, 256), color)
    ImageDraw.Draw(image).text((24, 116), label, fill="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def assert_text(name: str, response: dict) -> str:
    text = response["choices"][0]["message"]["content"]
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"{name}: empty response")
    print(f"{name}: {text[:160]!r}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    try:
        models = get_json(f"{args.base_url}/v1/models")["data"]
        model_ids = [entry["id"] for entry in models]
        if args.model not in model_ids:
            raise RuntimeError(f"served model missing: {model_ids}")
        assert_text(
            "text",
            chat(
                args.base_url,
                args.model,
                [{"type": "text", "text": "Reply with the word ready."}],
                False,
            ),
        )
        red = image_data_uri("#b91c1c", "RED")
        blue = image_data_uri("#1d4ed8", "BLUE")
        assert_text(
            "one-image",
            chat(
                args.base_url,
                args.model,
                [
                    {"type": "image_url", "image_url": {"url": red}},
                    {"type": "text", "text": "Name the dominant color."},
                ],
                False,
            ),
        )
        assert_text(
            "two-image",
            chat(
                args.base_url,
                args.model,
                [
                    {"type": "image_url", "image_url": {"url": red}},
                    {"type": "image_url", "image_url": {"url": blue}},
                    {"type": "text", "text": "Describe the colors in order."},
                ],
                False,
            ),
        )
        for index in range(3):
            text = assert_text(
                f"json-{index + 1}",
                chat(
                    args.base_url,
                    args.model,
                    [
                        {
                            "type": "text",
                            "text": (
                                "Return a JSON object with key status and value ok."
                            ),
                        }
                    ],
                    True,
                ),
            )
            if json.loads(text).get("status") != "ok":
                raise RuntimeError(f"json-{index + 1}: unexpected value")
    except Exception as error:
        print(f"phase135 smoke: FAIL: {error}", file=sys.stderr)
        raise
    print("phase135 smoke: PASS")


if __name__ == "__main__":
    main()
