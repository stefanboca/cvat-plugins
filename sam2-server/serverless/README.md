# Server-side SAM2 serverless function

## Purpose

This function runs all SAM2.1 inference on the server. It returns an RLE mask
to CVAT. The function ID is `pth-facebookresearch-sam2-server`.

The function has a per-worker image feature cache. The default cache size is
two images. Set `SAM2_IMAGE_CACHE_SIZE` during deployment to change the size.
A larger cache uses more GPU or system memory.

## Deploy the GPU function

Run this command from the `sam2-server` directory:

```console
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu.yaml \
  --platform local \
  --volume /absolute/host/path/checkpoints:/opt/nuclio/checkpoints \
  --env SAM2_CHECKPOINT=/opt/nuclio/checkpoints/sam2.1_hiera_tiny.pt
```

Use `function-gpu-cu126.yaml` for a compute capability 5.2 GPU. Use
`function.yaml` for a CPU.

## Select a GPU

Set `NVIDIA_VISIBLE_DEVICES` during deployment. For example, set the browser
function to GPU 0 and this function to GPU 1:

```console
# Run this command from the sam2 directory.
nuctl deploy --project-name cvat --path serverless/nuclio \
  --file serverless/nuclio/function-gpu-cu126.yaml --platform local \
  --volume /absolute/host/path/checkpoints:/opt/nuclio/checkpoints \
  --env SAM2_CHECKPOINT=/opt/nuclio/checkpoints/sam2.1_hiera_tiny.pt \
  --env NVIDIA_VISIBLE_DEVICES=0

# Run this command from the sam2-server directory.
nuctl deploy --project-name cvat --path serverless/nuclio \
  --file serverless/nuclio/function-gpu-cu126.yaml --platform local \
  --volume /absolute/host/path/checkpoints:/opt/nuclio/checkpoints \
  --env SAM2_CHECKPOINT=/opt/nuclio/checkpoints/sam2.1_hiera_tiny.pt \
  --env NVIDIA_VISIBLE_DEVICES=1
```

Use GPU UUIDs instead of indices if the device order can change. Run
`nvidia-smi -L` to get the UUIDs. Each selected physical GPU is `cuda:0`
inside its container.
