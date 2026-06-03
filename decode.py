import base64
import json
from io import BytesIO

from PIL import Image


def decode_transaction_to_image(json_path: str, output_path: str = "decoded.png"):
    with open(json_path) as f:
        body = json.load(f)

    raw = body["transaction"]
    data = base64.b64decode(raw)

    image = Image.open(BytesIO(data))
    image.save(output_path)
    print(f"saved: {output_path} | size: {image.size} | format: {image.format}")
    return image
