from typing import Any, Dict, List

from models import get_layout_model, get_ocr

PDF_RENDER_ZOOM = 2.0


def load_document(document: bytes) -> List[Dict[str, Any]]:
    """Decode raw bytes into a list of pages: {"index", "image"} (RGB)."""
    from io import BytesIO

    from PIL import Image

    if b"%PDF-" in document[:1024]:
        return _render_pdf(document)
    try:
        image = Image.open(BytesIO(document)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
        preview = document[:16].hex()
        raise ValueError(
            "decoded bytes are not a readable image or PDF "
            f"(len={len(document)}, first16=0x{preview}): {exc}"
        ) from exc
    return [{"index": 0, "image": image}]


def _render_pdf(document: bytes) -> List[Dict[str, Any]]:
    from io import BytesIO

    import pymupdf
    from PIL import Image

    pages: List[Dict[str, Any]] = []
    doc = pymupdf.open(stream=document, filetype="pdf")
    try:
        matrix = pymupdf.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
        for index, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix)
            image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
            pages.append({"index": index, "image": image})
    finally:
        doc.close()
    return pages


def detect_layout(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run Dolphin-1.5 layout detection on each page."""
    model = get_layout_model()
    layout = []
    for page in pages:
        result = model.detect(page["image"])
        result["page_index"] = page["index"]
        layout.append(result)
    print(f"[DEBUG] layout: {len(layout)} pages, elements: {[len(p.get('elements',[])) for p in layout]}")
    return layout


def _padded_box(bbox, width: int, height: int, pad: int):
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def crop_regions(
    pages: List[Dict[str, Any]],
    layout_json: List[Dict[str, Any]],
    padding: int = 28,
) -> List[Dict[str, Any]]:
    """Crop every detected bbox (padded) from its page image."""
    pages_by_index = {page["index"]: page["image"] for page in pages}
    crops: List[Dict[str, Any]] = []
    for layout in layout_json:
        page_index = layout.get("page_index", 0)
        image = pages_by_index.get(page_index)
        if image is None:
            continue
        width, height = image.size
        for element_index, element in enumerate(layout.get("elements", [])):
            bbox = element.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            box = _padded_box(bbox, width, height, padding)
            crops.append(
                {
                    "page_index": page_index,
                    "label": str(element.get("label", "region")),
                    "reading_order": int(element.get("reading_order", element_index)),
                    "bbox": bbox,
                    "padded_box": list(box),
                    "image": image.crop(box),
                }
            )
    print(f"[DEBUG] crops: {len(crops)}")
    return crops


def run_ocr(crops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run PaddleOCR on each crop, attaching ocr_text and text_lines."""
    import numpy as np

    ocr = get_ocr()
    rows: List[Dict[str, Any]] = []
    for crop in crops:
        # PaddleOCR consumes BGR ndarrays; PIL crops are RGB.
        array = np.asarray(crop["image"])[:, :, ::-1]
        try:
            text, lines = _ocr_array(ocr, array)
            ok, error = True, ""
        except Exception as exc:  # noqa: BLE001 - keep one bad crop from failing the page
            text, lines = "", []
            ok, error = False, repr(exc)
        print(f"[DEBUG] crop {crop['label']} reading_order={crop['reading_order']} text={repr(text[:100])}")
        rows.append(
            {
                "page_index": crop["page_index"],
                "label": crop["label"],
                "reading_order": crop["reading_order"],
                "bbox": crop["bbox"],
                "padded_box": crop["padded_box"],
                "ocr_text": text,
                "text_lines": lines,
                "ok": ok,
                "error": error,
            }
        )
    return rows


def _ocr_array(ocr, array):
    texts: List[str] = []
    lines: List[Dict[str, Any]] = []
    for result in ocr.predict(array):
        fields = _extract_ocr_fields(_result_to_dict(result))
        if fields["ocr_text"]:
            texts.append(fields["ocr_text"])
        lines.extend(fields["text_lines"])
    return "\n".join(texts), lines


def _result_to_dict(result: Any) -> Dict[str, Any]:
    payload = getattr(result, "json", None)
    if payload is not None:
        return _normalize_json(payload)
    if isinstance(result, dict):
        return _normalize_json(result)
    return {"raw_result": str(result)}


def _normalize_json(value: Any) -> Any:
    import json

    if isinstance(value, dict):
        return {k: _normalize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return value.tolist() if hasattr(value, "tolist") else str(value)


def _extract_ocr_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    res = payload.get("res", payload)
    texts = res.get("rec_texts") or []
    scores = res.get("rec_scores") or []
    boxes = res.get("rec_boxes") or res.get("rec_polys") or []
    lines = []
    for idx, text in enumerate(texts):
        line = {"text": str(text)}
        if idx < len(scores):
            line["score"] = scores[idx]
        if idx < len(boxes):
            line["box"] = boxes[idx]
        lines.append(line)
    return {"text_lines": lines, "ocr_text": "\n".join(str(t) for t in texts)}


def sort_reading_order(ocr_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep crops in layout reading order: page first, then reading_order."""
    return sorted(
        ocr_rows,
        key=lambda row: (row.get("page_index", 0), row.get("reading_order", 0)),
    )
