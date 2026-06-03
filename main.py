from fastapi import FastAPI
from helper import check_it_decode
from items import APIRequest, APIResponse
from parser import parse_fields
from pipeline import (
    crop_regions,
    detect_layout,
    load_document,
    run_ocr,
    sort_reading_order,
)

app = FastAPI()


@app.post("/ocr", response_model=APIResponse)
async def ocr(req: APIRequest):
    document = check_it_decode(req)
    pages = load_document(document)
    layout_json = detect_layout(pages)
    crops = crop_regions(
        pages=pages,
        layout_json=layout_json,
        padding=28,
    )
    ocr_rows = run_ocr(crops)

    ordered_rows = sort_reading_order(ocr_rows)

    pred_json = parse_fields(ordered_rows)

    return APIResponse(**pred_json)
