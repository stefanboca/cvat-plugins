import base64
import io
import json

from PIL import Image
from model_handler import ModelHandler


def init_context(context):
    context.logger.info("Initializing SAM2 context")
    context.user_data.model = ModelHandler()
    context.logger.info("SAM2 context initialized")


def handler(context, event):
    image = Image.open(io.BytesIO(base64.b64decode(event.body["image"]))).convert("RGB")
    blobs = context.user_data.model.handle(image)
    return context.Response(
        body=json.dumps({"version": 1, "dtype": "float16", "image_blobs": [base64.b64encode(blob).decode() for blob in blobs]}),
        headers={}, content_type="application/json", status_code=200,
    )
