# CVAT plugins

This repository contains external user-interface plugins for CVAT.

## Available plugins

### SAM3

The `sam3` directory contains a SAM3 interactive segmentation plugin. The
plugin uses a serverless image and text encoder. It runs the grounding decoder
in the browser.

Read `sam3/README.md` for the plugin operation and the server requirements.

## Build CVAT with plugins

`CLIENT_PLUGINS` is a build-time variable. Use an absolute path because this
repository is outside the CVAT source tree. Separate multiple paths with a
colon.

```console
CLIENT_PLUGINS=/absolute/path/to/plugin-a:/absolute/path/to/plugin-b \
yarn workspace cvat-ui run build
```

Read the README file for each plugin before you build it. A plugin can require
more variables or preparation steps.
