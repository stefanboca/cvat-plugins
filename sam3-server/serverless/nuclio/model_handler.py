import hashlib
import io
import os
import threading
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sam3.model import vitdet
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

from lru_cache import LRUCache


def addmm_act_fp32(activation, linear, inputs):
    outputs = linear(inputs)
    if activation in (F.relu, torch.nn.ReLU):
        return F.relu(outputs)
    if activation in (F.gelu, torch.nn.GELU):
        return F.gelu(outputs)
    raise ValueError(f"Unexpected activation {activation}")


def read_cache_size(name, default):
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def mask_to_rle(mask):
    y_coordinates, x_coordinates = np.nonzero(mask)
    if x_coordinates.size == 0:
        return [0, 0, 0, 0, 0]

    xtl = int(x_coordinates.min())
    ytl = int(y_coordinates.min())
    xbr = int(x_coordinates.max())
    ybr = int(y_coordinates.max())
    pixels = mask[ytl : ybr + 1, xtl : xbr + 1].reshape(-1).astype(np.uint8)
    changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    rle = np.diff(np.concatenate(([0], changes, [pixels.size]))).tolist()
    if pixels[0] == 1:
        rle.insert(0, 0)
    rle.extend([xtl, ytl, xbr, ybr])
    return rle


class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bfloat16 = (
            self.device.type == "cuda"
            and torch.cuda.get_device_capability()[0] >= 8
        )
        if self.device.type == "cuda" and not self.use_bfloat16:
            vitdet.addmm_act = addmm_act_fp32

        checkpoint_path = os.getenv("SAM3_CHECKPOINT")
        if checkpoint_path and not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"SAM3_CHECKPOINT does not exist: {checkpoint_path}"
            )

        self.model = build_sam3_image_model(
            device=self.device.type,
            eval_mode=True,
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        ).to(self.device)
        self.model.eval()
        self.processor = Sam3Processor(
            self.model, resolution=1008, device=self.device
        )
        self.image_cache = LRUCache(
            read_cache_size("SAM3_IMAGE_CACHE_SIZE", 2)
        )
        self.text_cache = LRUCache(
            read_cache_size("SAM3_TEXT_CACHE_SIZE", 32)
        )
        self._lock = threading.RLock()
        self._image_ids = torch.zeros(1, dtype=torch.long, device=self.device)
        self._text_ids = torch.zeros(1, dtype=torch.long, device=self.device)

    def _autocast(self):
        if self.use_bfloat16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _get_image_features(self, image_bytes):
        key = hashlib.sha256(image_bytes).digest()
        cached = self.image_cache.get(key)
        if cached is not None:
            return cached, True

        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
            width, height = image.size
            with self._autocast():
                state = self.processor.set_image(image)
        cached = (state["backbone_out"], width, height)
        self.image_cache.put(key, cached)
        return cached, False

    def _get_text_features(self, prompt):
        cached = self.text_cache.get(prompt)
        if cached is not None:
            return cached, True

        with self._autocast():
            cached = self.model.backbone.forward_text(
                [prompt], device=self.device
            )
        self.text_cache.put(prompt, cached)
        return cached, False

    def _make_prompt(self, pos_points, neg_points, obj_bbox, width, height):
        if obj_bbox and len(obj_bbox) != 2:
            raise ValueError("SAM3 requires two corners for one box")
        box_embeddings = None
        box_labels = None
        if obj_bbox:
            first, second = obj_bbox
            xtl = min(first[0], second[0]) / width
            ytl = min(first[1], second[1]) / height
            xbr = max(first[0], second[0]) / width
            ybr = max(first[1], second[1]) / height
            box_embeddings = torch.tensor(
                [
                    [
                        (xtl + xbr) / 2,
                        (ytl + ybr) / 2,
                        xbr - xtl,
                        ybr - ytl,
                    ]
                ],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1)
            box_labels = torch.ones(
                1, 1, dtype=torch.long, device=self.device
            )

        prompts = [(point, 1) for point in pos_points]
        prompts.extend((point, 0) for point in neg_points)
        point_embeddings = None
        point_labels = None
        if prompts:
            point_embeddings = torch.tensor(
                [
                    [[point[0] / width, point[1] / height]]
                    for point, _label in prompts
                ],
                dtype=torch.float32,
                device=self.device,
            )
            point_labels = torch.tensor(
                [[label] for _point, label in prompts],
                dtype=torch.long,
                device=self.device,
            )

        if box_embeddings is None and point_embeddings is None:
            return Prompt(
                box_embeddings=torch.zeros(
                    0, 1, 4, dtype=torch.float32, device=self.device
                ),
                box_mask=torch.zeros(
                    1, 0, dtype=torch.bool, device=self.device
                ),
            )

        return Prompt(
            box_embeddings=box_embeddings,
            box_labels=box_labels,
            point_embeddings=point_embeddings,
            point_labels=point_labels,
        )

    @torch.inference_mode()
    def handle(self, image_bytes, prompt, pos_points, neg_points, obj_bbox):
        with self._lock:
            image_entry, image_hit = self._get_image_features(image_bytes)
            image_features, width, height = image_entry
            text_features, text_hit = self._get_text_features(prompt)
            backbone_out = {**image_features, **text_features}
            geometric_prompt = self._make_prompt(
                pos_points, neg_points, obj_bbox, width, height
            )
            find_input = FindStage(
                img_ids=self._image_ids,
                text_ids=self._text_ids,
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            with self._autocast():
                output = self.model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_input,
                    find_target=None,
                    geometric_prompt=geometric_prompt,
                )
                best_index = output["pred_logits"].squeeze(-1).argmax(dim=1)
                logits = output["pred_masks"][
                    self._image_ids, best_index
                ].unsqueeze(1)
                logits = F.interpolate(
                    logits,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            mask = logits[0, 0].float().cpu().numpy() > 0
            return mask_to_rle(mask), image_hit, text_hit
