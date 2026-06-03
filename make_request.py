"""Build an /ocr request body from any image or PDF file.

Usage:
    python make_request.py path/to/image.png             # writes request.json
    python make_request.py path/to/doc.pdf out.json      # custom output name

Then POST it:
    curl -X POST http://localhost:8000/ocr \
         -H "Content-Type: application/json" \
         --data @request.json
"""

import base64
import json
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "request.json"

    with open(src, "rb") as fh:
        data = fh.read()

    body = {"header": "test", "transaction": base64.b64encode(data).decode()}
    with open(out, "w") as fh:
        json.dump(body, fh)

    print(f"wrote {out} ({len(data)} bytes -> {len(body['transaction'])} base64 chars)")


if __name__ == "__main__":
    main()
