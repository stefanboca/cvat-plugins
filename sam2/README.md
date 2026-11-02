# SAM 2.1 with browser decoding

Interactive segmentation in CVAT using `facebook/sam2.1-hiera-large`. A Nuclio
function computes image features, and the browser runs the decoder with ONNX
Runtime Web. WebGPU is preferred, with WebAssembly as a fallback.

## Setup

Run commands from the repository root unless noted otherwise.

1. Export the decoder from the checkpoint:

   ```sh
   SAM2_BUILD_CUDA=0 uv run sam2/scripts/export_decoder_onnx.py \
     --checkpoint /path/to/sam2.1_hiera_large.pt
   ```

   This writes `sam2/assets/sam2_decoder.onnx` and checks its output against
   PyTorch, including different prompt counts and coordinates.

2. [Deploy the matching Nuclio function](serverless/README.md).
3. Build CVAT with this plugin, running from the CVAT checkout:

   ```sh
   CLIENT_PLUGINS=/absolute/path/to/cvat-plugins/sam2 \
   CVAT_UI_ROOT=/absolute/path/to/cvat/cvat-ui \
   yarn workspace cvat-ui run build
   ```

The build copies the decoder into CVAT's assets. The server encoder and
browser decoder must use the same checkpoint.

## Development

Type-check from the plugin repository after installing CVAT's dependencies:

```sh
CVAT_UI_ROOT=/path/to/cvat/cvat-ui node sam2/scripts/typecheck.cjs
```

Set `localStorage["cvat.sam2.executionProvider"]` to `"wasm"` or `"webgpu"`
in the browser console, then reload, to select a decoder backend. Remove the
key to restore automatic selection. Set `localStorage["cvat.sam2.debug"]`
to `"1"` to log prompts and results.
