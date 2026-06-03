import os
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

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
    try:
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
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
