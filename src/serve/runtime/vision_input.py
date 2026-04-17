from __future__ import annotations

import base64
import binascii
import io
import os
from typing import List, Tuple

import torch
from PIL import Image, ImageFile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .env_utils import _env_bool
from .state import STATE

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_vision_transform_cache: Tuple[int, object] = (0, None)


def _vision_input_size() -> int:
    raw = os.environ.get("VISION_INPUT_SIZE")
    if raw:
        return int(raw)
    try:
        cfg = STATE.model.config  # type: ignore[union-attr]
        return int(getattr(cfg, "force_image_size", None) or cfg.vision_config.image_size)
    except Exception:
        return 448


def _build_vision_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _get_vision_transform():
    global _vision_transform_cache
    size = _vision_input_size()
    if _vision_transform_cache[0] != size:
        _vision_transform_cache = (size, _build_vision_transform(size))
    return _vision_transform_cache[1]


def _pixel_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _image_max_bytes() -> int:
    return int(os.environ.get("IMAGE_MAX_BYTES", str(25 * 1024 * 1024)))


def _fetch_image_bytes(url: str) -> bytes:
    url = url.strip()
    if url.startswith("data:"):
        if ";base64," not in url:
            raise ValueError("data: URL must include ;base64,")
        b64 = url.split(";base64,", 1)[1].strip()
        try:
            raw = base64.b64decode(b64, validate=False)
        except binascii.Error as e:
            raise ValueError("invalid base64 in data URL") from e
    elif url.startswith(("http://", "https://")):
        req = Request(url, headers={"User-Agent": "UniPercept-OpenAI-Server/1.0"})
        try:
            with urlopen(req, timeout=float(os.environ.get("IMAGE_FETCH_TIMEOUT_SEC", "60"))) as resp:
                raw = resp.read()
        except HTTPError as e:
            raise ValueError(f"failed to fetch image URL: HTTP {e.code}") from e
        except URLError as e:
            raise ValueError(f"failed to fetch image URL: {e.reason}") from e
    elif url.startswith("file://") and _env_bool("ALLOW_FILE_IMAGE_URL", False):
        path = url[7:]
        if path.startswith("//"):
            path = path[1:]
        workspace = os.path.realpath("/workspace")
        target = os.path.realpath(path)
        if not target.startswith(workspace + os.sep) and target != workspace:
            raise ValueError("file: URL must resolve under /workspace")
        with open(target, "rb") as f:
            raw = f.read()
    else:
        raise ValueError(
            "unsupported image URL (use data:...;base64,... or http(s)://, or enable ALLOW_FILE_IMAGE_URL for file:// under /workspace)"
        )
    if len(raw) > _image_max_bytes():
        raise ValueError(f"image exceeds IMAGE_MAX_BYTES ({_image_max_bytes()})")
    return raw


def _bytes_to_pixel_row(raw: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    t = _get_vision_transform()(img)
    return t


def _load_stacked_pixel_values_cpu(urls: List[str]) -> torch.Tensor:
    if not urls:
        raise ValueError("no image URLs")
    rows = [_bytes_to_pixel_row(_fetch_image_bytes(u)) for u in urls]
    return torch.stack(rows, dim=0)
