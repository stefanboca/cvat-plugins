import base64
import json

from model_handler import ModelHandler


def init_context(context):
    context.logger.info("Initializing server-side SAM3 context")
    context.user_data.model = ModelHandler()
    context.logger.info("Server-side SAM3 context initialized")


def handler(context, event):
    data = event.body
    text_prompts = data.get("text_prompts") or []
    prompt = (
        text_prompts[0].strip()
        if text_prompts and text_prompts[0].strip()
        else "visual"
    )
    image_bytes = base64.b64decode(data["image"], validate=True)
    rle, image_hit, text_hit = context.user_data.model.handle(
        image_bytes=image_bytes,
        prompt=prompt,
        pos_points=data.get("pos_points") or [],
        neg_points=data.get("neg_points") or [],
        obj_bbox=data.get("obj_bbox") or [],
    )
    context.logger.info(
        "SAM3 request completed: "
        f"image_cache_hit={image_hit}, text_cache_hit={text_hit}"
    )
    return context.Response(
        body=json.dumps(
            {
                "shapes": [
                    {
                        "points": rle,
                        "group": 0,
                        "source": "semi-auto",
                        "attributes": [],
                        "occluded": False,
                        "rotation": 0,
                        "type": "mask",
                    }
                ]
            }
        ),
        headers={},
        content_type="application/json",
        status_code=200,
    )
