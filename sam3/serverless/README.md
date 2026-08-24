# SAM3 serverless function

## Purpose

This function calculates image features and text features for the SAM3 UI
plugin. It sends FP16 data to the browser. The browser runs the grounding
decoder.

The function ID is `pth-facebookresearch-sam3`.

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
  --file serverless/nuclio/function-gpu.yaml \
  --platform local \
  --volume /absolute/host/HF_HUB_CACHE:/opt/nuclio/huggingface/hub \
  --env HF_HUB_OFFLINE=1
```

## Deploy the GPU function

Run this command from the `sam3` directory:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu.yaml \
  --platform local \
  --volume /absolute/host/HF_HUB_CACHE:/opt/nuclio/huggingface/hub \
  --env HF_HUB_OFFLINE=1
```

## Deploy the CPU function

Run this command from the `sam3` directory:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function.yaml \
  --platform local \
  --volume /absolute/host/HF_HUB_CACHE:/opt/nuclio/huggingface/hub \
  --env HF_HUB_OFFLINE=1
```

## Deploy the CUDA 12.6 GPU function

Run this command from the `sam3` directory:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu-cu126.yaml \
  --platform local \
  --volume /absolute/host/HF_HUB_CACHE:/opt/nuclio/huggingface/hub \
  --env HF_HUB_OFFLINE=1
```

## Select a GPU

Set `NVIDIA_VISIBLE_DEVICES` during deployment. For example, add
`--env NVIDIA_VISIBLE_DEVICES=0` to use physical GPU 0. Add
`--env NVIDIA_VISIBLE_DEVICES=1` to use physical GPU 1.

Use a GPU UUID for a stable selection. Run `nvidia-smi -L` to get the UUID.
The selected physical GPU is `cuda:0` inside the container.

The supplied YAML sets `HF_HUB_OFFLINE=1`. This setting prevents network
requests. If the cache is incomplete, deploy with `--env HF_HUB_OFFLINE=0`
and give the function an `HF_TOKEN`.

## Cache operation

The browser has separate caches for image features and text features. Each
request identifies the feature that is not in a cache. The function calculates
only that feature.

CVAT always includes an image in an interactor request. For a text-only
request, the function does not decode the image. It does not run the image
encoder.
