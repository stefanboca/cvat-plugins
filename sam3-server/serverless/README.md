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

## Get the checkpoint

The official SAM3 checkpoint is gated. Accept its license on Hugging Face.

Use one of these methods:

1. Give the function an `HF_TOKEN` when you deploy the function.
2. Put the checkpoint in the container. Set `SAM3_CHECKPOINT` to its absolute
   path.

The function downloads the checkpoint only when `SAM3_CHECKPOINT` is not set.

For example, use this command to mount a checkpoint from the host and deploy
the CUDA 12.6 function:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu-cu126.yaml \
  --platform local \
  --volume /absolute/host/path/checkpoints:/opt/nuclio/checkpoints \
  --env SAM3_CHECKPOINT=/opt/nuclio/checkpoints/sam3.pt
```

Replace `/absolute/host/path/checkpoints` with the absolute path of the host
directory that contains `sam3.pt`.

If `sam3.pt` is a symbolic link in a Hugging Face snapshot, mount the complete
model cache. The snapshot link needs the `blobs` directory. For example:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu-cu126.yaml \
  --platform local \
  --volume /var/cache/huggingface/hub/models--facebook--sam3:/opt/nuclio/checkpoints \
  --env SAM3_CHECKPOINT=/opt/nuclio/checkpoints/snapshots/REVISION/sam3.pt
```

Replace `REVISION` with the snapshot revision directory name.

## Deploy the GPU function

Run this command from the `sam3-server` directory:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu.yaml \
  --platform local --env HF_TOKEN=hf_...
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
  --env HF_TOKEN=hf_...
```

The image cache stores feature tensors on the inference device. A larger image
cache uses more GPU or system memory. The text features are smaller than the
image features.

Each function worker has separate caches. The supplied configurations use one
worker. A function restart clears both caches.

## Cache keys

The CVAT server does not send task, job, frame, or ROI identifiers to an
interactor function. The function uses a SHA-256 digest of the image payload
as the image cache key. The function uses the normalized text prompt as the
text cache key.
