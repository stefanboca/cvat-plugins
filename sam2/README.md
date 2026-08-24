# SAM2 plugin for CVAT

## Purpose

This plugin adds SAM2.1 interactive segmentation to the CVAT user interface.
The Nuclio function calculates image features. The browser runs the decoder
with ONNX Runtime Web. WebGPU is the first execution provider. WebAssembly is
the fallback provider.

To test the WebAssembly decoder, run this command in the browser console and
reload CVAT:

```javascript
localStorage.setItem("cvat.sam2.executionProvider", "wasm");
```

Set the value to `webgpu` to require WebGPU. Remove the setting to restore the
automatic selection:

```javascript
localStorage.removeItem("cvat.sam2.executionProvider");
```

To log the prompts and decoder results, run these commands and reload CVAT:

```javascript
localStorage.setItem("cvat.sam2.debug", "1");
location.reload();
```

Remove the setting after the test:

```javascript
localStorage.removeItem("cvat.sam2.debug");
```

The plugin uses `facebook/sam2.1-hiera-large`. The browser decoder has 64
point slots and one box. Static inputs make the ONNX model easier to run in a
browser. The server-side variant does not have this point limit.

## Build the UI

Set `CVAT_UI_ROOT` to the `cvat-ui` directory. Add this directory to
`CLIENT_PLUGINS`:

```console
CLIENT_PLUGINS=/path/to/cvat-plugins/sam2 \
CVAT_UI_ROOT=/path/to/cvat/cvat-ui \
yarn build
```

The prepare script copies `assets/sam2_decoder.onnx` to the CVAT asset
directory before webpack starts.

## Check the TypeScript files

```console
CVAT_UI_ROOT=/path/to/cvat/cvat-ui node sam2/scripts/typecheck.cjs
```

## Export the decoder

Run this command from the repository root:

```console
SAM2_BUILD_CUDA=0 uv run sam2/scripts/export_decoder_onnx.py \
  --checkpoint /path/to/sam2.1_hiera_large.pt
```

The exporter writes `sam2/assets/sam2_decoder.onnx`. It compares the ONNX
output with the PyTorch output.

## Deploy the function

Read `serverless/README.md`.
