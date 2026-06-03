import base64
import binascii
import re
from io import BytesIO

from fastapi import HTTPException
from PIL import Image

from items import APIRequest

_DATA_URI_PREFIX = re.compile(r"^data:[^;,]*(;[^;,]+)*;base64,", re.IGNORECASE)


def check_it_decode(req: APIRequest) -> bytes:
    raw = req.transaction or ""
    raw = _DATA_URI_PREFIX.sub("", raw.strip())
    raw = re.sub(r"\s+", "", raw)
    raw = raw.replace("-", "+").replace("_", "/")
    raw += "=" * (-len(raw) % 4)

    try:
        img_byte = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(
            status_code=400, detail=f"transaction is not valid base64: {e}"
        )

    if not img_byte:
        raise HTTPException(
            status_code=400, detail="transaction decoded to empty bytes"
        )
    is_pdf = img_byte[:4] == b"%PDF"
    is_image = False
    if not is_pdf:
        try:
            Image.open(BytesIO(img_byte)).verify()
            is_image = True
        except Exception:
            is_image = False

    if not is_pdf and not is_image:
        raise HTTPException(
            status_code=400,
            detail=f"transaction is not a valid image or PDF (first16=0x{img_byte[:16].hex()})",
        )
    return img_byte
