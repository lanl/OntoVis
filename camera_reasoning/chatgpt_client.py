import base64
import os
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_MODEL = "gpt-4o"


def _encode_image(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def ask_chatgpt(
    prompt: str,
    screenshot_path: str,
    target_image_path: Optional[str] = None,
    extra_images: Optional[List[Tuple[str, str]]] = None,
    reference_items: Optional[List[Tuple[str, str, str]]] = None,
    model: Optional[str] = None,
) -> str:
    """Send the camera-reasoning prompt and screenshot(s) to the OpenAI API and return the reply text.

    `extra_images` is an optional list of (label, image_path) pairs, sent after
    screenshot_path/target_image_path — each preceded by a small "[label]" text
    block so the model can refer back to images by name. Used for experiments that
    need to show many labeled images at once (e.g. comparing against every
    reference view), without changing the existing screenshot/target_image_path
    behavior at all.

    `reference_items` is an optional list of (label, image_path, description) triples,
    sent after extra_images. Unlike extra_images, each reference's image is placed
    immediately followed by its own description text, so the model sees them paired
    rather than all images first and all descriptions after:
        [label 1]
        <image 1>
        <description 1>
        [label 2]
        <image 2>
        <description 2>
        ...

    `model` falls back to the AI_MODEL env var, then DEFAULT_MODEL. The API base URL
    can be overridden via the AI_URL env var (e.g. to point at an OpenAI-compatible proxy).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the .env file in the project root "
            "(see .env.example), or export it in your shell before starting Jupyter."
        )

    resolved_model = model or os.environ.get("AI_MODEL") or DEFAULT_MODEL
    base_url = os.environ.get("AI_URL") or None

    content = [{"type": "text", "text": prompt}]

    screenshot_b64 = _encode_image(screenshot_path)
    content.append(
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
    )

    if target_image_path and Path(target_image_path).exists():
        target_b64 = _encode_image(target_image_path)
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_b64}"}}
        )

    if extra_images:
        for label, path in extra_images:
            if not Path(path).exists():
                continue
            content.append({"type": "text", "text": f"[{label}]"})
            image_b64 = _encode_image(path)
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            )

    if reference_items:
        for label, path, description in reference_items:
            if not Path(path).exists():
                continue
            content.append({"type": "text", "text": f"[{label}]"})
            image_b64 = _encode_image(path)
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            )
            content.append({"type": "text", "text": description})

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=resolved_model,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content
