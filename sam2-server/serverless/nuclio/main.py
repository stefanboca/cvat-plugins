import base64
import json

from model_handler import ModelHandler


def init_context(context):
    context.logger.info("Initializing server-side SAM2 context")
    context.user_data.model = ModelHandler()
    context.logger.info("Server-side SAM2 context initialized")


def handler(context, event):
    data = event.body
    rle, cache_hit = context.user_data.model.handle(
        base64.b64decode(data["image"]),
        data.get("pos_points") or [],
        data.get("neg_points") or [],
        data.get("obj_bbox") or [],
    )
    context.logger.info(f"SAM2 request completed: image_cache_hit={cache_hit}")
    return context.Response(
        body=json.dumps({"shapes": [{
            "points": rle, "group": 0, "source": "semi-auto",
            "attributes": [], "occluded": False, "rotation": 0, "type": "mask",
        }]}),
        headers={}, content_type="application/json", status_code=200,
    )
