# Server-side SAM2 plugin for CVAT

## Purpose

This directory contains a full server-side SAM2.1 interactor. The Nuclio
function runs the image encoder and the mask decoder. CVAT uses the function
as a normal dashboard interactor. No custom CVAT UI bundle is necessary.

The function uses the SAM2.1 Hiera tiny checkpoint. It uses the exact number
of supplied points. It does not add point slots or padding.

Read `serverless/README.md` for deployment instructions.
