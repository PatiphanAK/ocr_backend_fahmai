import base64

from fastapi import HTTPException
from items import APIRequest


def check_it_decode(req: APIRequest) -> bytes:
    try:
        return base64.b64decode(req.transaction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
