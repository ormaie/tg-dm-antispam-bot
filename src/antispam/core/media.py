from __future__ import annotations

import base64
import io
import logging

from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)


def compress_image_to_jpeg(
    raw_bytes: bytes,
    *,
    max_dimension: int = 768,
    quality: int = 75,
) -> bytes | None:
    """Decode an image/thumbnail, resize proportionally, and encode as compact JPEG."""
    if not raw_bytes:
        return None
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img = ImageOps.exif_transpose(img)
            # Extract first frame for animated GIF/WebP
            if getattr(img, "is_animated", False):
                img.seek(0)
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.split()[3])
                rgb = background
            elif img.mode != "RGB":
                rgb = img.convert("RGB")
            else:
                rgb = img.copy()

            if max(rgb.size) > max_dimension:
                rgb.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            out = io.BytesIO()
            rgb.save(out, format="JPEG", quality=quality, optimize=True)
            return out.getvalue()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.debug("Failed to process image bytes: %s", exc)
        return None


def bytes_to_data_url(jpeg_bytes: bytes) -> str:
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def is_high_risk_filename(file_name: str | None, high_risk_extensions: list[str]) -> bool:
    if not file_name:
        return False
    lowered = file_name.strip().lower()
    return any(lowered.endswith(ext.lower()) for ext in high_risk_extensions)
