# Server-side SAM3 serverless function

## Purpose

This function runs all SAM3 inference on the server. The function returns an
RLE mask to CVAT.

The function ID is `pth-facebookresearch-sam3-server`.

## Select a configuration

Use `nuclio/function-gpu.yaml` for an NVIDIA GPU.

Use `nuclio/function-gpu-cu126.yaml` for an NVIDIA GPU that has compute
capability 5.2. This configuration installs the CUDA 12.6 PyTorch packages.

Use `nuclio/function.yaml` for a CPU-only system. CPU inference is slow. Set
`CVAT_NUCLIO_DEFAULT_TIMEOUT` to 300 or more if CVAT stops the request before
the function completes.

## Prepare the model cache

The official SAM3 checkpoint is gated. Accept its license on Hugging Face.
Download `facebook/sam3` to your Hugging Face hub cache. Mount the complete
hub cache. It must include the `blobs`, `refs`, and `snapshots` directories.

For example:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu-cu126.yaml \
  --platform local \
  --volume /absolute/host/HF_HUB_CACHE:/opt/nuclio/huggingface/hub \
  --env HF_HUB_OFFLINE=1
```

## Deploy the GPU function

Run this command from the `sam3-server` directory:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu.yaml \
  --platform local \
  --volume /absolute/host/HF_HUB_CACHE:/opt/nuclio/huggingface/hub \
  --env HF_HUB_OFFLINE=1
```

## Set the cache sizes

The default image cache size is two. The default text cache size is 32. Set
the sizes during deployment if you need different limits:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu.yaml \
  --platform local \
  --env SAM3_IMAGE_CACHE_SIZE=1 \
  --env SAM3_TEXT_CACHE_SIZE=16 \
  --volume /absolute/host/HF_HUB_CACHE:/opt/nuclio/huggingface/hub
```

The image cache stores feature tensors on the inference device. A larger image
cache uses more GPU or system memory. The text features are smaller than the
image features.

Each function worker has separate caches. The supplied configurations use one
worker. A function restart clears both caches.

## Select a GPU

Set `NVIDIA_VISIBLE_DEVICES` during deployment. For example, add
`--env NVIDIA_VISIBLE_DEVICES=1` to use physical GPU 1. You can deploy the
browser-decoder function with `--env NVIDIA_VISIBLE_DEVICES=0` to put the two
functions on different GPUs.

Use a GPU UUID for a stable selection. Run `nvidia-smi -L` to get the UUID.
The selected physical GPU is `cuda:0` inside the container.

The supplied YAML sets `HF_HUB_OFFLINE=1`. This setting prevents network
requests. If the cache is incomplete, deploy with `--env HF_HUB_OFFLINE=0`
and give the function an `HF_TOKEN`.

## Cache keys

The CVAT server does not send task, job, frame, or ROI identifiers to an
interactor function. The function uses a SHA-256 digest of the image payload
as the image cache key. The function uses the normalized text prompt as the
text cache key.
