# SAM 3 with browser decoding

Interactive segmentation in CVAT using text, a box, and positive or negative
points. A Nuclio function computes image and text features; the browser runs
the grounding decoder with ONNX Runtime Web, using WebGPU or WebAssembly.
Image and text features are cached separately and transferred in FP16 format.

## Setup

1. Obtain the official SAM 3 checkpoint through Hugging Face.
2. Export the decoder from the `sam3` directory:

   ```sh
   uv run scripts/export_grounding_onnx.py \
     --checkpoint /path/to/sam3.pt \
     --validation-image /path/to/upstream-sam3/assets/images/truck.jpg
   ```

   This writes `assets/sam3_decoder.onnx` and checks it against the official
   PyTorch implementation, including FP16 feature transfer.

3. [Deploy the matching Nuclio function](serverless/README.md).
4. Build from the CVAT checkout:

   ```sh
   CLIENT_PLUGINS=/absolute/path/to/cvat-plugins/sam3 \
   CVAT_UI_ROOT=/absolute/path/to/cvat/cvat-ui \
   yarn workspace cvat-ui run build
   ```

`CVAT_UI_ROOT` is required: the prepare script copies the decoder into the
CVAT build. The server function ID is `pth-facebookresearch-sam3`.

## Limits and development

The exported decoder accepts one box, up to 64 points, and 32 text tokens, with
a 1008-by-1008 image input. An empty text prompt uses `visual`. Image and text
caches each hold 32 entries.

Type-check from the plugin repository after installing CVAT's dependencies:

```sh
CVAT_UI_ROOT=/path/to/cvat/cvat-ui node sam3/scripts/typecheck.cjs
```
