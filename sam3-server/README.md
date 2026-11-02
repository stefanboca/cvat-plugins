# SAM 3 with server-side inference

Runs the image encoder, text encoder, and grounding decoder on the server,
returning an RLE mask to CVAT. This is an alternative to the
[browser-decoder plugin](../sam3/README.md).

Prompts can contain text, one box, and positive or negative points. An empty
text prompt uses `visual`. Image and text features are cached so repeated
prompts only need to rerun the decoder.

## Setup

1. [Deploy the Nuclio function](serverless/README.md). Its ID is
   `pth-facebookresearch-sam3-server`.
2. Build CVAT with the plugin, running from the CVAT checkout:

   ```sh
   CLIENT_PLUGINS=/absolute/path/to/cvat-plugins/sam3-server \
   yarn workspace cvat-ui run build
   ```

Both SAM 3 variants can be included in one build by listing their absolute
paths in `CLIENT_PLUGINS`, separated by a colon. Add `CVAT_UI_ROOT` when
including the browser-decoder variant.

## Development

From the plugin repository, with CVAT's dependencies installed:

```sh
CVAT_UI_ROOT=/path/to/cvat/cvat-ui node sam3-server/scripts/typecheck.cjs
```

See the deployment guide for image and text cache settings.
