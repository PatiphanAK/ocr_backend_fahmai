import json
import os
import re

import cv2
import numpy as np
from PIL import Image
from vllm import SamplingParams
from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt, TokensPrompt

OCR_DEVICE = os.environ.get("OCR_DEVICE", "gpu:0")
OCR_DET_MODEL = os.environ.get("OCR_DET_MODEL", "PP-OCRv5_mobile_det")
OCR_REC_MODEL = os.environ.get("OCR_REC_MODEL", "th_PP-OCRv5_mobile_rec")
OCR_BATCH_SIZE = int(os.environ.get("OCR_BATCH_SIZE", "16"))
OCR_DET_LIMIT_SIDE_LEN = int(os.environ.get("OCR_DET_LIMIT_SIDE_LEN", "960"))

# 28x28 - 1 = 783 (Dolphin encoder patch count, see vLLM example)
ENCODER_PROMPT_STATIC = "".join(["0"] * 783)
SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=2048)
LAYOUT_PROMPT = "Parse the reading order of this document. "

_ocr = None


def _prepare_image(image):
    """Pad to square (Dolphin expects square input). Returns padded BGR + dims."""
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = image_cv.shape[:2]
    max_size = max(h, w)
    top = (max_size - h) // 2
    bottom = max_size - h - top
    left = (max_size - w) // 2
    right = max_size - w - left
    padded = cv2.copyMakeBorder(
        image_cv, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return padded, {
        "orig_w": w,
        "orig_h": h,
        "pad_w": max_size,
        "pad_h": max_size,
        "top": top,
        "left": left,
    }


def _parse_layout_string(bbox_str):
    """Parse '[x1,y1,x2,y2] label' entries (normalized coords)."""
    pattern = r"\[(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+)\]\s*(\w+)"
    results = []
    for m in re.finditer(pattern, bbox_str):
        coords = [float(m.group(i)) for i in range(1, 5)]
        results.append((coords, m.group(5).strip()))
    return results


def _to_original_bbox(coords, dims):
    """Normalized coords on padded image -> pixel coords on original image."""
    x1 = int(coords[0] * dims["pad_w"]) - dims["left"]
    y1 = int(coords[1] * dims["pad_h"]) - dims["top"]
    x2 = int(coords[2] * dims["pad_w"]) - dims["left"]
    y2 = int(coords[3] * dims["pad_h"]) - dims["top"]
    x1 = max(0, min(x1, dims["orig_w"]))
    y1 = max(0, min(y1, dims["orig_h"]))
    x2 = max(0, min(x2, dims["orig_w"]))
    y2 = max(0, min(y2, dims["orig_h"]))
    if x2 <= x1:
        x2 = min(x1 + 1, dims["orig_w"])
    if y2 <= y1:
        y2 = min(y1 + 1, dims["orig_h"])
    return [float(x1), float(y1), float(x2), float(y2)]


def run_dolphin_vllm(image, vllm_engine, processor) -> dict:
    width, height = image.size

    decoder_prompt = f"<s>{LAYOUT_PROMPT}<Answer/>"
    decoder_tokens = TokensPrompt(
        prompt_token_ids=processor.tokenizer(decoder_prompt, add_special_tokens=False)[
            "input_ids"
        ]
    )
    enc_dec = ExplicitEncoderDecoderPrompt(
        encoder_prompt=TextPrompt(
            prompt=ENCODER_PROMPT_STATIC, multi_modal_data={"image": image}
        ),
        decoder_prompt=decoder_tokens,
    )

    outputs = vllm_engine.generate(prompts=enc_dec, sampling_params=SAMPLING_PARAMS)
    decoded = outputs[0].outputs[0].text

    _, dims = _prepare_image(image)
    parsed = _parse_layout_string(decoded)

    elements = []
    for order, (coords, label) in enumerate(parsed):
        if label == "fig":
            continue
        elements.append(
            {
                "label": label,
                "bbox": _to_original_bbox(coords, dims),
                "reading_order": order,
            }
        )

    return {"image_width": width, "image_height": height, "elements": elements}


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
