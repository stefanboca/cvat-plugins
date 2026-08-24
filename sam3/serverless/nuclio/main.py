import base64
import io
import json

from model_handler import ModelHandler
from PIL import Image


def init_context(context):
    context.logger.info("Initializing SAM3 context")
    context.user_data.model = ModelHandler()
    context.logger.info("SAM3 context initialized")


def handler(context, event):
    data = event.body
    text_prompts = data.get("text_prompts") or []
    prompt = (
        text_prompts[0].strip()
        if text_prompts and text_prompts[0].strip()
        else "visual"
    )
    request_marker = text_prompts[1] if len(text_prompts) > 1 else ""
    marker_parts = (
        request_marker.split(":") if request_marker.startswith("cvat-sam3:") else []
    )
    requests = {
        "image": not marker_parts or "image" in marker_parts[1:],
        "text": not marker_parts or "text" in marker_parts[1:],
    }
    image = None
    if requests["image"]:
        image = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")
    result = context.user_data.model.handle(image, prompt, requests)

    return context.Response(
        body=json.dumps({"version": 1, "dtype": "float16", **result}),
        headers={},
        content_type="application/json",
        status_code=200,
    )
