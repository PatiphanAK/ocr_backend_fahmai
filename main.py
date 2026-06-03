import os
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return where the error came from instead of a bare Internal Server Error."""
    tb = traceback.extract_tb(exc.__traceback__)
    last = tb[-1] if tb else None
    where = (
        f"{last.filename}:{last.lineno} in {last.name}() -> {last.line}"
        if last is not None
        else "unknown"
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "message": str(exc),
            "where": where,
            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
        },
    )


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
