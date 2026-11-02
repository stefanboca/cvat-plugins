# CVAT segmentation plugins

SAM 2.1 and SAM 3 integrations for interactive image annotation in CVAT. They
let users guide segmentation with points and, for SAM 3, text and boxes.

The repository contains CVAT UI plugins, model-export tools, and Nuclio
functions for serving the models. Each model has a browser-decoder variant
and a version that runs inference entirely on the server.

| Model | Browser decoder | Server-side inference |
| --- | --- | --- |
| SAM 2.1 | [sam2](sam2/README.md) | [sam2-server](sam2-server/README.md) |
| SAM 3 | [sam3](sam3/README.md) | [sam3-server](sam3-server/README.md) |

## Build a UI plugin

From the CVAT checkout, after preparing the model as described in the linked
plugin guide:

```sh
CLIENT_PLUGINS=/absolute/path/to/cvat-plugins/sam3 \
CVAT_UI_ROOT=/absolute/path/to/cvat/cvat-ui \
yarn workspace cvat-ui run build
```

The plugin guides cover model preparation and deployment. The server-side
SAM 2 function works with CVAT's standard interactor interface.
