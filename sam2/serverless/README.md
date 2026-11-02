# SAM 2.1 function deployment

Computes three FP16 image feature maps for the browser decoder.
The function ID is `pth-facebookresearch-sam2`.

## Model and configuration

Download `facebook/sam2.1-hiera-large` to your Hugging Face cache.
Mount the complete hub cache, including `blobs`, `refs`, and `snapshots`.

| Configuration in `nuclio/` | Runtime |
| --- | --- |
| `function-gpu.yaml` | NVIDIA GPU |
| `function-gpu-cu126.yaml` | CUDA 12.6 packages for compute capability 5.2 |
| `function.yaml` | CPU |

## Deploy

Run from the `sam2` directory, replacing the cache path:

```sh
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu.yaml \
  --platform local \
  --volume /absolute/host/HF_HUB_CACHE:/opt/nuclio/huggingface/hub \
  --env HF_HUB_OFFLINE=1
```

Select another YAML file for CPU or CUDA 12.6. With an incomplete cache, allow
downloads with `--env HF_HUB_OFFLINE=0`.

## GPU selection and caching

Add `--env CUDA_VISIBLE_DEVICES=0` to select a GPU, or use a UUID from
`nvidia-smi -L` for stable device selection. The selected GPU appears as
`cuda:0` inside the container. To restrict container access, change
`nvidia.com/gpu=all` in `spec.devices` to a device such as `nvidia.com/gpu=0`.

The server encoder and browser decoder must use the same model weights.
Export a matching decoder when changing the checkpoint.
