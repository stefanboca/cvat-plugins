#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "einops",
#     "ml-dtypes>=0.5.1",
#     "numpy<2",
#     "onnx",
#     "onnxruntime-gpu",
#     "onnxscript",
#     "pillow",
#     "psutil",
#     "pycocotools",
#     "sam3",
#     "setuptools<82.0.0",
#     "torch",
# ]
# [tool.uv.sources]
# sam3 = { git = "https://github.com/facebookresearch/sam3", rev = "8f0b7f4d4e7eda2ed606ebde6702c93359ad01da" }
# ///

"""Export the ONNX decoder used by CVAT's SAM3 UI plugin.

The serverless function computes image and text backbones. This graph accepts
their three image FPN tensors plus the 32-token contextual text representation
and evaluates SAM3 grounding for one padded box and 64 padded points. Keeping
language features as inputs lets the browser cache text and image embeddings
independently.

Inputs use these fixed shapes:

* ``fpn_288``: float32 ``[1, 256, 288, 288]``
* ``fpn_144``: float32 ``[1, 256, 144, 144]``
* ``fpn_72``: float32 ``[1, 256, 72, 72]``
* ``boxes`` / ``box_valid`` / ``box_labels``: ``[1, 1, 4]``, ``[1, 1]``, ``[1, 1]``
* ``points`` / ``point_valid`` / ``point_labels``: ``[1, 64, 2]``, ``[1, 64]``, ``[1, 64]``
* ``language_features``: float32 ``[32, 1, 256]``
* ``language_mask``: bool ``[1, 32]``

The output is the highest-scoring candidate's raw ``[1, 1, 288, 288]`` mask
logits. The script converts internal weights to FP16 while retaining float32
I/O, checks the graph, and compares ONNX Runtime output with official PyTorch.
"""

from __future__ import annotations

import argparse
import gc
import math
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnxruntime.transformers.float16 import convert_float_to_float16
from PIL import Image
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from torch import Tensor, nn

INPUT_NAMES = [
    "fpn_288",
    "fpn_144",
    "fpn_72",
    "boxes",
    "box_valid",
    "box_labels",
    "points",
    "point_valid",
    "point_labels",
    "language_features",
    "language_mask",
]
OUTPUT_NAMES = ["mask_logits"]
OUTPUT_TOLERANCES = {"mask_logits": (1e-2, 1e-1)}
FP16_TRANSPORT_INPUTS = {
    "fpn_288",
    "fpn_144",
    "fpn_72",
    "language_features",
}


class GroundingONNXWrapper(nn.Module):
    """Tensor-only ONNX boundary around ``Sam3Image.forward_grounding``."""

    def __init__(self, model: nn.Module, vision_pos_72: Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("vision_pos_72", vision_pos_72.detach().clone())
        self.register_buffer(
            "img_ids", torch.zeros(1, dtype=torch.long, device=model.device)
        )
        self.register_buffer(
            "text_ids", torch.zeros(1, dtype=torch.long, device=model.device)
        )

    def forward(
        self,
        fpn_288: Tensor,
        fpn_144: Tensor,
        fpn_72: Tensor,
        boxes: Tensor,
        box_valid: Tensor,
        box_labels: Tensor,
        points: Tensor,
        point_valid: Tensor,
        point_labels: Tensor,
        language_features: Tensor,
        language_mask: Tensor,
    ) -> tuple[Tensor]:
        backbone_out = {
            "backbone_fpn": [fpn_288, fpn_144, fpn_72],
            "vision_pos_enc": [self.vision_pos_72],
            "language_features": language_features,
            "language_mask": language_mask,
        }
        geometric_prompt = Prompt(
            box_embeddings=boxes.transpose(0, 1),
            box_mask=~box_valid,
            box_labels=box_labels.transpose(0, 1),
            point_embeddings=points.transpose(0, 1),
            point_mask=~point_valid,
            point_labels=point_labels.transpose(0, 1),
        )
        find_input = FindStage(
            img_ids=self.img_ids,
            text_ids=self.text_ids,
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        output = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )
        best_index = output["pred_logits"].squeeze(-1).argmax(dim=1)
        return (output["pred_masks"][self.img_ids, best_index].unsqueeze(1),)


def parse_args() -> argparse.Namespace:
    plugin_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=plugin_root / "assets/sam3_decoder.onnx",
        help="Destination ONNX file (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional local sam3.pt; otherwise the official checkpoint is downloaded",
    )
    parser.add_argument(
        "--validation-image",
        type=Path,
        default=Path.cwd() / "assets/images/truck.jpg",
        help="Image used for mandatory PyTorch/ONNX validation",
    )
    parser.add_argument("--text", default="visual", help="Validation text prompt")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.validation_image.is_file():
        raise FileNotFoundError(f"validation image not found: {args.validation_image}")


@torch.no_grad()
def make_wrapper(model: nn.Module, device: torch.device) -> nn.Module:
    dummy_72 = torch.zeros(1, 256, 72, 72, device=device)
    vision_pos_72 = model.backbone.vision_backbone.position_encoding(dummy_72)
    return GroundingONNXWrapper(model, vision_pos_72).eval()


@torch.no_grad()
def make_example_inputs(
    model: nn.Module,
    device: torch.device,
    image_path: Path,
    text: str,
) -> tuple[Tensor, ...]:
    processor = Sam3Processor(model, resolution=1008, device=device)
    with Image.open(image_path) as image:
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = processor.set_image(image.convert("RGB"))
        else:
            state = processor.set_image(image.convert("RGB"))

    fpn_288, fpn_144, fpn_72 = (
        value.detach().float().clone()
        for value in state["backbone_out"]["backbone_fpn"]
    )
    text_output = model.backbone.forward_text([text], device=device)
    language_features = text_output["language_features"].detach().float().clone()
    language_mask = text_output["language_mask"].detach().clone()

    boxes = torch.zeros(1, 1, 4, device=device)
    boxes[:, 0] = torch.tensor([0.5, 0.46875, 0.9166667, 0.4791667], device=device)
    box_valid = torch.ones(1, 1, dtype=torch.bool, device=device)
    box_labels = torch.ones(1, 1, dtype=torch.long, device=device)
    points = torch.zeros(1, 64, 2, device=device)
    point_valid = torch.zeros(1, 64, dtype=torch.bool, device=device)
    point_labels = torch.zeros(1, 64, dtype=torch.long, device=device)
    return (
        fpn_288,
        fpn_144,
        fpn_72,
        boxes,
        box_valid,
        box_labels,
        points,
        point_valid,
        point_labels,
        language_features,
        language_mask,
    )


@contextmanager
def sam3_export_workarounds() -> Iterator[None]:
    """Select graph-safe SAM3 paths only while torch.export captures."""

    original_pin_memory = torch.Tensor.pin_memory
    original_is_dynamo_compiling = torch.compiler.is_dynamo_compiling
    original_sdpa = F.scaled_dot_product_attention

    def identity_pin_memory(tensor: Tensor, device: object = None) -> Tensor:
        del device
        return tensor

    def math_sdpa(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> Tensor:
        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), dim=-3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), dim=-3)
        scale_factor = scale if scale is not None else 1.0 / math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        if is_causal:
            causal_mask = torch.ones(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).tril()
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        weights = torch.softmax(scores, dim=-1)
        if dropout_p:
            weights = torch.dropout(weights, dropout_p, train=True)
        return torch.matmul(weights, value)

    torch.Tensor.pin_memory = identity_pin_memory  # type: ignore[method-assign]
    torch.compiler.is_dynamo_compiling = lambda: True
    F.scaled_dot_product_attention = math_sdpa
    try:
        yield
    finally:
        torch.Tensor.pin_memory = original_pin_memory  # type: ignore[method-assign]
        torch.compiler.is_dynamo_compiling = original_is_dynamo_compiling
        F.scaled_dot_product_attention = original_sdpa


def export_fp32(
    wrapper: nn.Module,
    inputs: Sequence[Tensor],
    output: Path,
) -> tuple[np.ndarray, ...]:
    with torch.inference_mode():
        source_outputs = wrapper(*inputs)
    source_arrays = tuple(
        value.detach().float().cpu().numpy() for value in source_outputs
    )
    with sam3_export_workarounds():
        torch.onnx.export(
            wrapper,
            tuple(inputs),
            str(output),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamo=True,
            external_data=True,
        )
    return source_arrays


def topologically_sort_graph(graph: onnx.GraphProto) -> None:
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                topologically_sort_graph(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    topologically_sort_graph(child)

    nodes = list(graph.node)
    produced = {name for node in nodes for name in node.output if name}
    available = {value.name for value in graph.input}
    available.update(initializer.name for initializer in graph.initializer)
    available.update(
        name for node in nodes for name in node.input if name and name not in produced
    )
    ordered: list[onnx.NodeProto] = []
    pending = nodes
    while pending:
        ready = [
            node
            for node in pending
            if all(not name or name in available for name in node.input)
        ]
        if not ready:
            unresolved = sorted(
                name
                for node in pending
                for name in node.input
                if name and name not in available
            )
            raise ValueError(f"could not topologically sort ONNX graph: {unresolved}")
        ready_ids = {id(node) for node in ready}
        pending = [node for node in pending if id(node) not in ready_ids]
        ordered.extend(ready)
        available.update(name for node in ready for name in node.output if name)
    del graph.node[:]
    graph.node.extend(ordered)


def reduce_to_fp16(fp32_model: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(fp32_model), load_external_data=True)
    model = convert_float_to_float16(model, keep_io_types=True)
    topologically_sort_graph(model.graph)
    onnx.save_model(model, str(output), save_as_external_data=False)


def verify(
    output: Path,
    inputs: dict[str, np.ndarray],
    expected: Sequence[np.ndarray],
) -> None:
    available = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else []
    )
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(output), providers=providers)
    actual = session.run(OUTPUT_NAMES, inputs)
    failures = []
    for name, source, result in zip(OUTPUT_NAMES, expected, actual):
        difference = np.abs(result.astype(np.float32) - source)
        max_abs = float(difference.max(initial=0.0))
        denominator = np.maximum(np.abs(source), 1e-6)
        max_rel = float((difference / denominator).max(initial=0.0))
        rtol, atol = OUTPUT_TOLERANCES[name]
        close = np.isclose(result, source, rtol=rtol, atol=atol, equal_nan=False)
        mismatch = 100.0 * (1.0 - float(close.mean()))
        print(
            f"{name}: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}, "
            f"outside_tolerance={mismatch:.4g}%"
        )
        if source.shape != result.shape or not bool(close.all()):
            failures.append(name)
    if failures:
        raise AssertionError(f"ONNX outputs differ from PyTorch: {', '.join(failures)}")


def main() -> None:
    args = parse_args()
    check_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        load_from_HF=args.checkpoint is None,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    ).to(device)
    model.eval()
    wrapper = make_wrapper(model, device)
    inputs = make_example_inputs(model, device, args.validation_image, args.text)

    with tempfile.TemporaryDirectory(prefix="sam3-onnx-export-") as directory:
        fp32_output = Path(directory) / "decoder.fp32.onnx"
        source_outputs = export_fp32(wrapper, inputs, fp32_output)
        reduce_to_fp16(fp32_output, args.output)

    onnx.checker.check_model(str(args.output))
    size_mib = args.output.stat().st_size / (1024 * 1024)
    print(f"Checked {args.output} ({size_mib:.1f} MiB)")
    ort_inputs = {}
    for name, value in zip(INPUT_NAMES, inputs):
        array = value.detach().cpu().numpy()
        # Match the Nuclio-to-browser wire format used in production. The
        # worker expands these FP16 blobs back to float32 before ONNX Runtime.
        if name in FP16_TRANSPORT_INPUTS:
            array = array.astype(np.float16).astype(np.float32)
        ort_inputs[name] = array
    del inputs, wrapper, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    verify(args.output, ort_inputs, source_outputs)


if __name__ == "__main__":
    main()
