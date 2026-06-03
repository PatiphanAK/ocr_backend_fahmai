import base64
import binascii
import re

from fastapi import HTTPException
from items import APIRequest

_DATA_URI_PREFIX = re.compile(r"^data:[^;,]*(;[^;,]+)*;base64,", re.IGNORECASE)


def check_it_decode(req: APIRequest) -> bytes:
    raw = req.transaction or ""

    # Drop a data-URI prefix (e.g. "data:image/png;base64,") if present.
    raw = _DATA_URI_PREFIX.sub("", raw.strip())
    # Remove any whitespace/newlines the client may have inserted.
    raw = re.sub(r"\s+", "", raw)
    # Normalise URL-safe base64 (-/_ -> +//) back to the standard alphabet.
    raw = raw.replace("-", "+").replace("_", "/")
    # Restore missing padding.
    raw += "=" * (-len(raw) % 4)

    try:
        # validate=True so malformed input raises instead of silently
        # discarding bytes and producing an unreadable "image".
        document = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"transaction is not valid base64: {e}")

    if not document:
        raise HTTPException(status_code=400, detail="transaction decoded to empty bytes")

    return document
