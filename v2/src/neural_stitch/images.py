"""Loading and resizing of input images."""
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class InputImage:
    name: str
    rgb: np.ndarray  # [H, W, 3] uint8

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)"""
        return self.rgb.shape[1], self.rgb.shape[0]


def decode_image(data: bytes | Path | str, name: str, max_side: int) -> InputImage:
    """Decode any Pillow-readable image: applies EXIF rotation, drops alpha, caps the long side."""
    src = io.BytesIO(data) if isinstance(data, bytes) else data
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        rgb = np.asarray(im)
    rgb, _ = resize_max(rgb, max_side)
    return InputImage(name=name, rgb=np.ascontiguousarray(rgb))


def load_folder(folder: Path | str, max_side: int) -> list[InputImage]:
    """Load all images of a folder, sorted by name (the order does not matter for stitching)."""
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [decode_image(p, p.stem, max_side) for p in paths]


def resize_max(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Shrink so the long side is at most max_side. Returns the image and the scale factor."""
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s == 1.0:
        return rgb, 1.0
    size = (max(1, round(w * s)), max(1, round(h * s)))
    # Use the exact factor that was applied along the long side.
    s = size[0] / w if w >= h else size[1] / h
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), s


def thumbnail_jpeg(rgb: np.ndarray, side: int = 256) -> bytes:
    small, _ = resize_max(rgb, side)
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def save_png(rgba: np.ndarray, path: Path | str) -> None:
    Image.fromarray(rgba).save(path, format="PNG")
