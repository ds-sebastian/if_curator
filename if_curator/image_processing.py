"""Image processing functions for cropping faces and objects."""

import logging
import os

from PIL import Image

logger = logging.getLogger(__name__)

# Lazy singleton
_yolo_model = None


def get_yolo_model():
    """Singleton for YOLO model."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        logger.info("Loading YOLOv9c model...")
        _yolo_model = YOLO("yolov9c.pt")
    return _yolo_model


def process_face_mode(
    img: Image.Image,
    asset: dict,
    person: dict,
    output_dir: str,
    count: int,
    min_width: int = 50,
) -> bool:
    """Crop face based on Immich metadata and save to output directory."""
    # Find face metadata for this person
    face_info = None
    for p in asset.get("people", []):
        if p["id"] == person["id"] and (faces := p.get("faces")):
            face_info = faces[0]
            break

    if not face_info:
        logger.debug(f"No face info for {person.get('name')} in asset {asset.get('id')}")
        return False

    img_w, img_h = img.size
    meta_w = face_info.get("imageWidth") or img_w
    meta_h = face_info.get("imageHeight") or img_h

    # Scale bounding box to actual image dimensions
    scale_x, scale_y = img_w / meta_w, img_h / meta_h
    x1 = face_info["boundingBoxX1"] * scale_x
    y1 = face_info["boundingBoxY1"] * scale_y
    x2 = face_info["boundingBoxX2"] * scale_x
    y2 = face_info["boundingBoxY2"] * scale_y

    face_w, face_h = x2 - x1, y2 - y1
    if face_w < min_width or face_h < min_width:
        logger.debug(f"Face too small ({face_w:.1f}x{face_h:.1f})")
        return False

    # Add 10% margin
    margin_x, margin_y = face_w * 0.10, face_h * 0.10
    crop_box = (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(img_w, x2 + margin_x),
        min(img_h, y2 + margin_y),
    )

    face_crop = img.crop(crop_box)
    face_crop.save(os.path.join(output_dir, f"{count}.jpg"), format="JPEG")
    return True


def process_object_mode(
    img: Image.Image,
    config: dict,
    output_dir: str,
    count: int,
) -> bool:
    """Detect and crop objects using YOLO."""
    try:
        model = get_yolo_model()
        target_class = config.get("object_class", "dog")
        device = "cpu" if os.getenv("FORCE_CPU", "").lower() in ("true", "1", "yes") else None

        results = model(img, verbose=False, device=device)

        found = False
        for idx, (box, cls_id, conf) in enumerate(
            (box, int(box.cls[0]), float(box.conf[0]))
            for r in results
            for box in r.boxes
        ):
            if 0 <= cls_id < len(model.names) and model.names[cls_id] == target_class and conf > 0.5:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                img.crop((x1, y1, x2, y2)).save(
                    os.path.join(output_dir, f"{count}_{idx}.jpg"),
                    format="JPEG",
                )
                found = True

        return found
    except Exception as e:
        logger.error(f"YOLO processing failed: {e}")
        return False


def process_full_mode(img: Image.Image, output_dir: str, count: int) -> bool:
    """Save full image."""
    img.save(os.path.join(output_dir, f"{count}.jpg"), format="JPEG")
    return True
