from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


INTERPOLATION = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "nearest": Image.Resampling.NEAREST,
}


def load_rgb_image(payload: bytes) -> Image.Image:
    """Decode uploaded bytes into a RGB PIL image."""
    try:
        with Image.open(BytesIO(payload)) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a readable image.") from exc


def preprocess_classifier_image(
    image: Image.Image,
    *,
    crop_size: int,
    resize_size: int,
    resize_mode: str,
    interpolation: str,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """Preprocess a PIL image into NCHW float32 tensor for ONNX classifiers."""
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Uploaded image has invalid dimensions.")

    resample = INTERPOLATION.get(interpolation, Image.Resampling.BILINEAR)
    if resize_mode == "stretch":
        prepared = image.resize((crop_size, crop_size), resample)
    else:
        resized = resize_shortest_side(image, resize_size, resample)
        prepared = center_crop(resized, crop_size)

    array = np.asarray(prepared, dtype=np.float32) / 255.0
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (array - mean_array) / std_array
    chw = normalized.transpose(2, 0, 1)
    return np.expand_dims(chw, axis=0).astype(np.float32, copy=False)


def resize_shortest_side(image: Image.Image, resize_size: int, resample: Image.Resampling) -> Image.Image:
    width, height = image.size
    scale = resize_size / min(width, height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    return image.resize((new_width, new_height), resample)


def center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    left = max((width - crop_size) // 2, 0)
    top = max((height - crop_size) // 2, 0)
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))
