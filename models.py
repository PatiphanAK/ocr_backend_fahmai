import json
import os
import re

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

DOLPHIN_MODEL_PATH = os.environ.get("DOLPHIN_MODEL_PATH", "ByteDance/Dolphin")
DOLPHIN_ALPHA = float(os.environ.get("DOLPHIN_ALPHA", "0.25"))

OCR_DEVICE = os.environ.get("OCR_DEVICE", "gpu:0")
OCR_DET_MODEL = os.environ.get("OCR_DET_MODEL", "PP-OCRv5_mobile_det")
OCR_REC_MODEL = os.environ.get("OCR_REC_MODEL", "th_PP-OCRv5_mobile_rec")
OCR_BATCH_SIZE = int(os.environ.get("OCR_BATCH_SIZE", "16"))
OCR_DET_LIMIT_SIDE_LEN = int(os.environ.get("OCR_DET_LIMIT_SIDE_LEN", "960"))

_layout_model = None
_ocr = None


class DolphinLayout:
    def __init__(self, model_path: str, alpha: float):
        self.alpha = alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def detect(self, image) -> dict:
        import torch

        width, height = image.size
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=4096)
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return {
            "image_width": width,
            "image_height": height,
            "elements": _parse_layout_elements(decoded, width, height),
        }


def _parse_layout_elements(decoded: str, width: int, height: int) -> list:

    match = re.search(r"[\[{].*[\]}]", decoded, re.S)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    raw = payload.get("elements", payload) if isinstance(payload, dict) else payload
    elements = []
    for order, item in enumerate(raw if isinstance(raw, list) else []):
        bbox = item.get("bbox") or item.get("box")
        if not bbox or len(bbox) != 4:
            continue
        elements.append(
            {
                "label": str(item.get("label", "region")),
                "bbox": [float(v) for v in bbox],
                "reading_order": int(item.get("reading_order", order)),
            }
        )
    return elements


def get_layout_model() -> DolphinLayout:
    global _layout_model
    if _layout_model is None:
        _layout_model = DolphinLayout(DOLPHIN_MODEL_PATH, DOLPHIN_ALPHA)
    return _layout_model


def get_ocr():
    global _ocr
    if _ocr is None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        _ocr = PaddleOCR(
            ocr_version="PP-OCRv5",
            lang="th",
            device=OCR_DEVICE,
            text_detection_model_name=OCR_DET_MODEL,
            text_recognition_model_name=OCR_REC_MODEL,
            text_recognition_batch_size=OCR_BATCH_SIZE,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=OCR_DET_LIMIT_SIDE_LEN,
            text_det_limit_type="max",
        )
    return _ocr
