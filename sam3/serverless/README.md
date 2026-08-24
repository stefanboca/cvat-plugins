# SAM3 serverless function

## Purpose

This function calculates image features and text features for the SAM3 UI
plugin. It sends FP16 data to the browser. The browser runs the grounding
decoder.

The function ID is `pth-facebookresearch-sam3`.

## Select a configuration

Use `nuclio/function-gpu.yaml` for an NVIDIA GPU.

Use `nuclio/function.yaml` for a CPU-only system. CPU inference is slow. Set
`CVAT_NUCLIO_DEFAULT_TIMEOUT` to 300 or more if CVAT stops the request before
the function completes.

## Get the checkpoint

The official SAM3 checkpoint is gated. Accept its license on Hugging Face.

Use one of these methods:

1. Give the function an `HF_TOKEN` when you deploy it.
2. Put the checkpoint in the container. Set `SAM3_CHECKPOINT` to its absolute
   path.

The function downloads the checkpoint only when `SAM3_CHECKPOINT` is not set.

For example, use this command to mount a checkpoint from the host and deploy
the GPU function:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu.yaml \
  --platform local \
  --volume /absolute/host/path/checkpoints:/opt/nuclio/checkpoints \
  --env SAM3_CHECKPOINT=/opt/nuclio/checkpoints/sam3.pt
```

Replace `/absolute/host/path/checkpoints` with the absolute path of the host
directory that contains `sam3.pt`. `SAM3_CHECKPOINT` must contain the path in
the container.

## Deploy the GPU function

Run this command from the `sam3` directory:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu.yaml \
  --platform local --env HF_TOKEN=hf_...
```

## Deploy the CPU function

Run this command from the `sam3` directory:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function.yaml \
  --platform local --env HF_TOKEN=hf_...
```

## Cache operation

The browser has separate caches for image features and text features. Each
request identifies the feature that is not in a cache. The function calculates
only that feature.

CVAT always includes an image in an interactor request. For a text-only
request, the function does not decode the image. It does not run the image
encoder.
