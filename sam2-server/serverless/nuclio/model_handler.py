import hashlib
import io
import os
import threading
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from lru_cache import LRUCache


MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def mask_to_rle(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return [0, 0, 0, 0, 0]
    xtl, ytl, xbr, ybr = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    pixels = mask[ytl : ybr + 1, xtl : xbr + 1].reshape(-1).astype(np.uint8)
    changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    rle = np.diff(np.concatenate(([0], changes, [pixels.size]))).tolist()
    if pixels[0]:
        rle.insert(0, 0)
    rle.extend([xtl, ytl, xbr, ybr])
    return rle


class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = os.getenv("SAM2_CHECKPOINT")
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"SAM2_CHECKPOINT does not exist: {checkpoint}")
        config = os.getenv("SAM2_MODEL_CONFIG", MODEL_CONFIG)
        model = build_sam2(config, checkpoint, device=self.device, mode="eval")
        self.predictor = SAM2ImagePredictor(model)
        capacity = int(os.getenv("SAM2_IMAGE_CACHE_SIZE", "2"))
        if capacity < 1:
            raise ValueError("SAM2_IMAGE_CACHE_SIZE must be at least 1")
        self.image_cache = LRUCache(capacity)
        self.use_bfloat16 = self.device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        self._lock = threading.RLock()

    def _autocast(self):
        if self.use_bfloat16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _set_image(self, image_bytes):
        key = hashlib.sha256(image_bytes).digest()
        cached = self.image_cache.get(key)
        if cached is not None:
            self.predictor._features, self.predictor._orig_hw = cached
            self.predictor._is_image_set = True
            self.predictor._is_batch = False
            return True
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
            with self._autocast():
                self.predictor.set_image(image)
        self.image_cache.put(key, (self.predictor._features, self.predictor._orig_hw))
        return False

    def handle(self, image_bytes, pos_points, neg_points, obj_bbox):
        if obj_bbox and len(obj_bbox) != 2:
            raise ValueError("SAM2 requires two corners for one box")
        prompts = [(point, 1) for point in pos_points]
        prompts.extend((point, 0) for point in neg_points)
        coordinates = np.asarray([point for point, _ in prompts], dtype=np.float32) if prompts else None
        labels = np.asarray([label for _, label in prompts], dtype=np.int32) if prompts else None
        box = None
        if obj_bbox:
            first, second = obj_bbox
            box = np.asarray([
                min(first[0], second[0]), min(first[1], second[1]),
                max(first[0], second[0]), max(first[1], second[1]),
            ], dtype=np.float32)
        if coordinates is None and box is None:
            raise ValueError("SAM2 requires a box or a point prompt")
        with self._lock, self._autocast():
            cache_hit = self._set_image(image_bytes)
            masks, scores, _ = self.predictor.predict(
                point_coords=coordinates, point_labels=labels, box=box,
                multimask_output=True,
            )
        return mask_to_rle(masks[int(np.argmax(scores))]), cache_hit
