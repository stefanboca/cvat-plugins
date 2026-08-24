# SAM2 serverless function

## Purpose

This function calculates three FP16 image feature maps for the SAM2 browser
plugin. The function ID is `pth-facebookresearch-sam2`.

## Prepare the model cache

Download `facebook/sam2.1-hiera-large` to your Hugging Face hub cache. Mount
the complete directory in the container. The directory must include the
`blobs`, `refs`, and `snapshots` directories.

## Deploy the GPU function

Run this command from the `sam2` directory:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu.yaml \
  --platform local \
  --volume /absolute/host/HF_HUB_CACHE:/opt/nuclio/huggingface/hub \
  --env HF_HUB_OFFLINE=1
```

Use `function-gpu-cu126.yaml` for a compute capability 5.2 GPU. This file
installs the CUDA 12.6 PyTorch packages. Use `function.yaml` for a CPU.

## Select a GPU

Set `CUDA_VISIBLE_DEVICES` during deployment. The GPU YAML makes all GPUs
available to the container. This variable tells CUDA which GPU to use. For
example, this command uses physical GPU 0:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu-cu126.yaml \
  --platform local \
  --volume /absolute/host/HF_HUB_CACHE:/opt/nuclio/huggingface/hub \
  --env HF_HUB_OFFLINE=1 \
  --env CUDA_VISIBLE_DEVICES=0
```

Use a GPU UUID for a stable selection. Run `nvidia-smi -L` to get the UUID.
The selected physical GPU is `cuda:0` inside the container.

For container-level isolation, change `nvidia.com/gpu=all` in `spec.devices`
to a device such as `nvidia.com/gpu=0`. Do not set this field to `all` if the
container must not have access to the other GPUs.

The supplied YAML sets `HF_HUB_OFFLINE=1`. This setting prevents network
requests. If the cache is incomplete, deploy with `--env HF_HUB_OFFLINE=0`.

The function calls
`SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")`. It then
moves the model to the selected device. The browser ONNX decoder contains
weights from the same model. Do not change the model independently. Export a
matching decoder if you change it.
