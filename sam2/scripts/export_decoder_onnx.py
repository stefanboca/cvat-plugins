# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "ml-dtypes>=0.5.1", "numpy<2", "onnx", "onnxruntime", "onnxscript", "setuptools<82", "torch>=2.5.1", "torchvision>=0.20.1",
#   "sam-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4",
# ]
# [tool.uv.extra-build-dependencies]
# sam-2 = ["setuptools"]
# ///
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from sam2.build_sam import build_sam2


MAX_POINTS = 64


class Decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.prompt_encoder = model.sam_prompt_encoder
        self.mask_decoder = model.sam_mask_decoder

    def forward(self, high_res_0, high_res_1, image_embed, point_coords, point_labels):
        coordinates = point_coords.to(image_embed.dtype)
        sparse, dense = self.prompt_encoder(points=(coordinates, point_labels), boxes=None, masks=None)
        sparse = sparse.to(image_embed.dtype)
        dense = dense.to(image_embed.dtype)
        masks, scores, _, _ = self.mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.prompt_encoder.get_dense_pe().to(image_embed.dtype),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=False,
            high_res_features=[high_res_0, high_res_1],
        )
        return masks.float(), scores.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "assets/sam2_decoder.onnx")
    args = parser.parse_args()
    model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(args.checkpoint), device="cpu", mode="eval")
    decoder = Decoder(model).half().eval()
    # The official coordinate encoder always calculates its Fourier features
    # in FP32. Keep its matrix in FP32, then cast its output for the FP16 decoder.
    decoder.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix = (
        decoder.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix.float()
    )
    inputs = (
        torch.randn(1, 32, 256, 256, dtype=torch.float16),
        torch.randn(1, 64, 128, 128, dtype=torch.float16),
        torch.randn(1, 256, 64, 64, dtype=torch.float16),
        torch.zeros(1, MAX_POINTS + 2, 2, dtype=torch.float32),
        torch.full((1, MAX_POINTS + 2), -1, dtype=torch.int64),
    )
    inputs[4][0, 2] = 1
    inputs[3][0, 2] = torch.tensor([512.0, 512.0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        decoder, inputs, str(args.output), opset_version=18, dynamo=False,
        input_names=["high_res_0", "high_res_1", "image_embed", "point_coords", "point_labels"],
        output_names=["masks", "scores"],
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(args.output))
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    feeds = {item.name: value.numpy() for item, value in zip(session.get_inputs(), inputs)}
    expected = decoder(*inputs)
    actual = session.run(None, feeds)
    for name, reference, result in zip(("masks", "scores"), expected, actual):
        reference_array = reference.detach().numpy()
        np.testing.assert_allclose(result, reference_array, atol=0.25, rtol=0.03)
        difference = float(np.max(np.abs(result - reference_array)))
        print(f"Validated {name}: shape={result.shape}, max_abs_difference={difference:.6f}")
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
