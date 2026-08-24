import os

import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = os.getenv("SAM2_CHECKPOINT")
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"SAM2_CHECKPOINT does not exist: {checkpoint}")
        config = os.getenv("SAM2_MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")
        self.predictor = SAM2ImagePredictor(
            build_sam2(config, checkpoint, device=self.device, mode="eval")
        )
        self.use_bfloat16 = self.device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8

    @torch.inference_mode()
    def handle(self, image):
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.use_bfloat16 else __import__("contextlib").nullcontext()
        )
        with context:
            self.predictor.set_image(image)
        features = [
            *self.predictor._features["high_res_feats"],
            self.predictor._features["image_embed"],
        ]
        return [
            np.ascontiguousarray(feature.float().cpu().numpy().astype(np.float16)).tobytes()
            for feature in features
        ]
