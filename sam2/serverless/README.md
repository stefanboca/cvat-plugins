# SAM2 serverless function

## Purpose

This function calculates three FP16 image feature maps for the SAM2 browser
plugin. The function ID is `pth-facebookresearch-sam2`.

## Get the checkpoint

Download `sam2.1_hiera_tiny.pt` from the official SAM2 repository. Put the
file in a host directory.

## Deploy the GPU function

Run this command from the `sam2` directory:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu.yaml \
  --platform local \
  --volume /absolute/host/path/checkpoints:/opt/nuclio/checkpoints \
  --env SAM2_CHECKPOINT=/opt/nuclio/checkpoints/sam2.1_hiera_tiny.pt
```

Use `function-gpu-cu126.yaml` for a compute capability 5.2 GPU. This file
installs the CUDA 12.6 PyTorch packages. Use `function.yaml` for a CPU.

## Select a GPU

Set `NVIDIA_VISIBLE_DEVICES` during deployment. For example, this command uses
physical GPU 0:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu-cu126.yaml \
  --platform local \
  --volume /absolute/host/path/checkpoints:/opt/nuclio/checkpoints \
  --env SAM2_CHECKPOINT=/opt/nuclio/checkpoints/sam2.1_hiera_tiny.pt \
  --env NVIDIA_VISIBLE_DEVICES=0
```

Use a GPU UUID for a stable selection. Run `nvidia-smi -L` to get the UUID.
The selected physical GPU is `cuda:0` inside the container.
