# SAM3 plugin for CVAT

## Purpose

This plugin adds SAM3 interactive segmentation to the CVAT user interface.

The plugin accepts these prompts:

- One optional text prompt
- One optional box
- A maximum of 64 positive and negative points

If the text field is empty, the plugin uses the text `visual`.

## Operation

A Nuclio function calculates the SAM3 image features and text features. The
browser runs the SAM3 grounding decoder with ONNX Runtime Web.

The browser uses WebGPU when WebGPU is available. It uses WebAssembly when
WebGPU is not available. The browser keeps the image features and text
features in FP16 format. This format decreases conversion time and memory use.

The browser has two separate caches. The image cache and the text cache each
contain 32 entries.

If both entries are in the caches, the browser does not call the function. If
one entry is not in a cache, the function calculates only that entry.

CVAT always includes an image in an interactor request. For a text-only
request, the function does not decode the image. It does not run the image
encoder.

## Build CVAT with the plugin

Set the plugin path and the CVAT UI path when you build CVAT:

```console
CLIENT_PLUGINS=/path/to/cvat-plugins/sam3 \
CVAT_UI_ROOT=/path/to/cvat/cvat-ui \
yarn workspace cvat-ui run build
```

The build requires `CVAT_UI_ROOT`. Set it to the writable CVAT UI build
directory. The `prepare.cjs` script copies the decoder to
`dist/assets/sam3_decoder.onnx` before webpack starts.

Add the plugin to the Nix derivation before you build CVAT.

## Check TypeScript

Install the CVAT JavaScript dependencies first. Set `CVAT_UI_ROOT` to the CVAT
UI source directory. Run the check from the plugin repository:

```console
CVAT_UI_ROOT=/path/to/cvat/cvat-ui node sam3/scripts/typecheck.cjs
```

The script uses the CVAT TypeScript version and configuration. It checks the
CVAT UI and the external SAM3 source files. It does not copy the plugin into
CVAT.

## Generate the decoder

Run the exporter from the `sam3` directory:

```console
uv run scripts/export_grounding_onnx.py \
  --checkpoint /path/to/sam3.pt \
  --validation-image /path/to/upstream-sam3/assets/images/truck.jpg
```

The exporter writes `assets/sam3_decoder.onnx`. It compares the ONNX result
with the official PyTorch result. It also checks the FP16 data transfer that
the Nuclio function and browser use.

The decoder has these fixed limits:

- One box
- 64 points
- 32 text tokens
- A SAM3 input image size of 1008 by 1008 pixels

## Server requirement

Deploy an interactor with this function ID:

```text
pth-facebookresearch-sam3
```

The function must use the SAM3 feature protocol that this plugin expects. The
matching CPU and GPU configurations are in `serverless/nuclio`.

Read `serverless/README.md` for the deployment instructions.

The official model is gated. Accept its license on Hugging Face. Download the
model to a Hugging Face hub cache and mount the complete cache in the
container.
