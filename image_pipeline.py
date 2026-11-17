import io
import re
import base64
import logging
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageOps
import httpx

from config_loader import ImageLocalConfig, ImageCloudConfig

logger = logging.getLogger("gateway.images")

DATA_URI_PATTERN = re.compile(r"^data:image\/[a-zA-Z0-9\+\-\.]+;base64,(.+)$", re.DOTALL)


def has_images(messages: List[Dict[str, Any]]) -> bool:
    """Проверяет, содержит ли входящий список сообщений изображения."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


async def fetch_image_bytes(url: str, timeout_sec: float = 10.0) -> bytes:
    """Загружает изображение по URL или декодирует из Base64 Data URI."""
    match = DATA_URI_PATTERN.match(url)
    if match:
        b64_data = match.group(1).strip()
        return base64.b64decode(b64_data)

    if url.startswith("http://") or url.startswith("https://"):
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    raise ValueError(f"Unsupported image URL format: {url[:50]}...")


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (255, 255, 255)


def process_image_bytes(
    raw_bytes: bytes,
    max_side_px: int,
    output_format: str = "JPEG",
    quality: int = 85,
    convert_transparent_to_bg: str = "#FFFFFF",
    grayscale: bool = False,
) -> Tuple[bytes, str]:
    """
    Обрабатывает изображение в памяти:
    1. Замена альфа-канала на сплошной фон.
    2. Пропорциональный ресайз до max_side_px.
    3. Опциональный перевод в grayscale.
    4. Сжатие в JPEG/WEBP.
    """
    with Image.open(io.BytesIO(raw_bytes)) as img:
        # Автоповорот по EXIF
        img = ImageOps.exif_transpose(img)

        # Конвертация альфа-канала PNG/WEBP
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            bg_color = hex_to_rgb(convert_transparent_to_bg)
            background = Image.new("RGB", img.size, bg_color)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[3]) # 3 — alpha-channel
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        if grayscale:
            img = ImageOps.grayscale(img).convert("RGB")

        # Пропорциональный ресайз
        w, h = img.size
        max_dim = max(w, h)
        if max_dim > max_side_px:
            scale = max_side_px / float(max_dim)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Сохранение в буфер
        out_buf = io.BytesIO()
        fmt = output_format.upper()
        if fmt not in ("JPEG", "WEBP"):
            fmt = "JPEG"

        save_kwargs: Dict[str, Any] = {"quality": quality, "optimize": True}
        if fmt == "JPEG":
            save_kwargs["progressive"] = True

        img.save(out_buf, format=fmt, **save_kwargs)
        out_bytes = out_buf.getvalue()
        mime_type = "image/jpeg" if fmt == "JPEG" else "image/webp"
        return out_bytes, mime_type


async def process_messages_images(
    messages: List[Dict[str, Any]],
    target_backend: str,
    local_cfg: ImageLocalConfig,
    cloud_cfg: ImageCloudConfig,
    master_enabled: bool = True,
) -> List[Dict[str, Any]]:
    """
    Проходит по всем сообщениям и заменяет картинки на оптимизированные base64 data-uri
    в соответствии с профилем (local или cloud).
    """
    if not master_enabled:
        return messages

    cfg = local_cfg if target_backend == "local" else cloud_cfg
    if not cfg.enabled:
        return messages

    max_side = cfg.max_side_px
    quality = cfg.quality
    output_format = getattr(cfg, "output_format", "JPEG")
    bg_color = getattr(cfg, "convert_transparent_to_bg", "#FFFFFF")
    grayscale = getattr(cfg, "grayscale", False)

    new_messages: List[Dict[str, Any]] = []

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue

        new_content_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                img_obj = part.get("image_url", {})
                url = img_obj.get("url") if isinstance(img_obj, dict) else str(img_obj)
                if not url:
                    new_content_parts.append(part)
                    continue

                try:
                    raw_bytes = await fetch_image_bytes(url)
                    processed_bytes, mime = process_image_bytes(
                        raw_bytes=raw_bytes,
                        max_side_px=max_side,
                        output_format=output_format,
                        quality=quality,
                        convert_transparent_to_bg=bg_color,
                        grayscale=grayscale,
                    )
                    b64_str = base64.b64encode(processed_bytes).decode("ascii")
                    new_url = f"data:{mime};base64,{b64_str}"

                    new_part = dict(part)
                    if isinstance(img_obj, dict):
                        new_img_obj = dict(img_obj)
                        new_img_obj["url"] = new_url
                        new_part["image_url"] = new_img_obj
                    else:
                        new_part["image_url"] = {"url": new_url}
                    new_content_parts.append(new_part)
                except Exception as e:
                    logger.error(f"Failed to preprocess image: {e}. Keeping original.")
                    new_content_parts.append(part)
            else:
                new_content_parts.append(part)

        new_msg = dict(msg)
        new_msg["content"] = new_content_parts
        new_messages.append(new_msg)

    return new_messages
