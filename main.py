import os
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import DonutProcessor
from vllm import LLM, SamplingParams

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

load_dotenv()


app = FastAPI()
MODEL_ID = "ByteDance/Dolphin"
processor = DonutProcessor.from_pretrained(MODEL_ID)
vllm_engine = LLM(
    model=MODEL_ID,
    dtype="bfloat16",
    max_num_seqs=8,
    gpu_memory_utilization=0.5,
    hf_overrides={"architectures": ["DonutForConditionalGeneration"]},
)
ENCODER_PROMPT_STATIC = "".join(["0"] * 783)
SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=2048)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
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


@app.get("/health_be3")
def health_check():
    return JSONResponse(content={"status": "healthy"}, status_code=200)


@app.post("/ocr", response_model=APIResponse)
async def ocr(req: APIRequest):
    img_byte = check_it_decode(req)
    pages = load_document(img_byte)
    layout_json = detect_layout(pages, vllm_engine, processor)
    crops = crop_regions(
        pages=pages,
        layout_json=layout_json,
        padding=28,
    )
    ocr_rows = run_ocr(crops)
    ordered_rows = sort_reading_order(ocr_rows)
    for r in ordered_rows:
        print(f"[{r.label}] {repr(r.ocr_text)}")
    bank_statement = parse_fields(ordered_rows)
    return APIResponse(id=req.id, answer=bank_statement)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "28000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
