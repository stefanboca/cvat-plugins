import base64
import os

import torch
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = os.getenv("SAM3_CHECKPOINT")
        if checkpoint_path and not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"SAM3_CHECKPOINT does not exist: {checkpoint_path}"
            )

        self.model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
            enable_segmentation=False,
            enable_inst_interactivity=False,
            compile=False,
        )
        self.processor = Sam3Processor(self.model, resolution=1008, device=self.device)

    @torch.inference_mode()
    def encode_text(self, prompt):
        output = self.model.backbone.forward_text([prompt], device=self.device)
        features = (
            output["language_features"]
            .detach()
            .to(device="cpu", dtype=torch.float16)
            .contiguous()
            .numpy()
            .tobytes()
        )
        mask = (
            output["language_mask"]
            .detach()
            .to(device="cpu", dtype=torch.uint8)
            .contiguous()
            .numpy()
            .tobytes()
        )
        return {
            "prompt": prompt,
            "feature_blob": base64.b64encode(features).decode("ascii"),
            "mask_blob": base64.b64encode(mask).decode("ascii"),
        }

    @torch.inference_mode()
    def encode_image(self, image):
        if self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = self.processor.set_image(image)
        else:
            state = self.processor.set_image(image)

        features = state["backbone_out"]["backbone_fpn"]
        image_blobs = [
            base64.b64encode(
                feature.detach()
                .to(device="cpu", dtype=torch.float16)
                .contiguous()
                .numpy()
                .tobytes()
            ).decode("ascii")
            for feature in features
        ]
        return image_blobs

    def handle(self, image, prompt, requests):
        return {
            "image_blobs": self.encode_image(image) if requests["image"] else None,
            "text": self.encode_text(prompt) if requests["text"] else None,
        }
