# SAM 2.1 with server-side inference

Runs the image encoder and mask decoder on the server using
`facebook/sam2.1-hiera-large`. The function caches image features and returns
an RLE mask to CVAT. It uses the supplied points without padding.

This works as a standard CVAT dashboard interactor; no custom UI build is
needed. The function ID is `pth-facebookresearch-sam2-server`.

See [deployment instructions](serverless/README.md) for model caching, CPU
and GPU configurations, and cache settings.
