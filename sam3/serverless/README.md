# SAM 3 function deployment

Computes image and text features for the browser grounding decoder.
The function ID is `pth-facebookresearch-sam3`.

## Model and configuration

Accept the model license on Hugging Face and download `facebook/sam3`.
Mount the complete hub cache, including `blobs`, `refs`, and `snapshots`.

| Configuration in `nuclio/` | Runtime |
| --- | --- |
| `function-gpu.yaml` | NVIDIA GPU |
| `function-gpu-cu126.yaml` | CUDA 12.6 packages for compute capability 5.2 |
| `function.yaml` | CPU |

## Deploy

Run from the `sam3` directory, replacing the cache path:

```sh
nuctl deploy --project-name cvat \
  --path serverless/nuclio \
  --file serverless/nuclio/function-gpu.yaml \
  --platform local \
  --volume /absolute/host/HF_HUB_CACHE:/opt/nuclio/huggingface/hub \
  --env HF_HUB_OFFLINE=1
```

Select another YAML file for CPU or CUDA 12.6. With an incomplete cache, allow
downloads with `--env HF_HUB_OFFLINE=0` and supply an `HF_TOKEN`.
CPU inference can need `CVAT_NUCLIO_DEFAULT_TIMEOUT=300` or higher in CVAT.

## GPU selection and caching

Add `--env CUDA_VISIBLE_DEVICES=0` to select a GPU, or use a UUID from
`nvidia-smi -L` for stable device selection. The selected GPU appears as
`cuda:0` inside the container. To restrict container access, change
`nvidia.com/gpu=all` in `spec.devices` to a device such as `nvidia.com/gpu=0`.

The browser caches image and text features separately. The function computes
only features missing from those caches; text-only requests skip image
decoding and the image encoder.
