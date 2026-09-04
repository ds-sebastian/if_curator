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
            (box, int(box.cls[0]), float(box.conf[0])) for r in results for box in r.boxes
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
