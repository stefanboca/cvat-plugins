# Server-side SAM3 plugin for CVAT

## Purpose

This plugin adds a server-side SAM3 interactor to the CVAT user interface.
Use this plugin to compare server-side inference with the browser decoder in
the `sam3` plugin.

The plugin accepts these prompts:

- One optional text prompt
- One optional box
- Positive and negative points

If the text field is empty, the plugin uses the text `visual`.

## Operation

The Nuclio function runs the image encoder, the text encoder, and the grounding
decoder. The function sends an RLE mask to CVAT. The browser does not run an
ONNX model.

The function has two bounded LRU caches. The image cache contains two entries
by default. The text cache contains 32 entries by default. A repeated request
for the same image and text does not run either encoder. The grounding decoder
runs for each request because the box and point prompts can change.

The function uses a digest of the image payload as the image cache key. The
digest also identifies an ROI crop. The function uses the normalized text as
the text cache key.

Each Nuclio worker has separate caches. The supplied configurations use one
worker. Do not increase the worker count if you need one shared cache.

## Build CVAT with the plugin

Set the plugin path when you build CVAT:

```console
CLIENT_PLUGINS=/path/to/cvat-plugins/sam3-server \
yarn workspace cvat-ui run build
```

You can build both SAM3 plugins into the same CVAT UI:

```console
CLIENT_PLUGINS=/path/to/cvat-plugins/sam3:/path/to/cvat-plugins/sam3-server \
CVAT_UI_ROOT=/path/to/cvat/cvat-ui \
yarn workspace cvat-ui run build
```

The `sam3` browser-decoder plugin requires `CVAT_UI_ROOT`. The server-side
plugin does not require this variable during the build.

## Check TypeScript

Install the CVAT JavaScript dependencies first. Set `CVAT_UI_ROOT` to the CVAT
UI source directory. Run the check from the plugin repository:

```console
CVAT_UI_ROOT=/path/to/cvat/cvat-ui \
node sam3-server/scripts/typecheck.cjs
```

## Server requirement

Deploy an interactor with this function ID:

```text
pth-facebookresearch-sam3-server
```

This ID is different from the ID of the browser-decoder function. You can
deploy both functions in the same CVAT project.

Read `serverless/README.md` for the deployment instructions and cache
settings.
