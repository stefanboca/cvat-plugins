import { InferenceSession, Tensor, env } from "@modules/onnxruntime-web/dist/ort.webgpu.min.mjs";

const FEATURE_SHAPES = [[1, 32, 256, 256], [1, 64, 128, 128], [1, 256, 64, 64]] as const;
const MAX_POINTS = 64;
const PROMPT_SLOTS = MAX_POINTS + 2;
const MASK_SIZE = 256;
const MAX_CACHED_IMAGES = 2;
let decoder: InferenceSession | null = null;
const imageCache = new Map<string, Uint16Array[]>();
const resizeCache = new Map<number, { lower: Uint16Array; upper: Uint16Array; weight: Float32Array }>();
env.wasm.wasmPaths = "/assets/";

export type ExecutionProvider = "auto" | "webgpu" | "wasm";

async function createDecoder(url: string, provider: ExecutionProvider): Promise<InferenceSession> {
  if (provider !== "wasm" && "gpu" in navigator) {
    try {
      return await InferenceSession.create(url, {
        executionProviders: ["webgpu", "wasm"],
      });
    } catch (error) {
      if (provider === "webgpu") throw error;
      console.warn("SAM2 WebGPU initialization failed. Using WebAssembly.", error);
    }
  }
  return InferenceSession.create(url, { executionProviders: ["wasm"] });
}

export enum WorkerAction { INIT = "init", DECODE = "decode" }
export interface InitBody { decoderURL: string; executionProvider: ExecutionProvider }
export interface DecodeBody {
  imageKey: string;
  imageBlobs: [string, string, string] | null;
  coordinates: Float32Array;
  labels: BigInt64Array;
  width: number;
  height: number;
}
export interface SAM2OutputItem { points: Uint8ClampedArray; bounds: [number, number, number, number] }
interface WorkerInput { action: WorkerAction; requestID: number; payload: InitBody | DecodeBody }

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

interface Base64Uint8ArrayConstructor extends Uint8ArrayConstructor {
  fromBase64?: (value: string) => Uint8Array;
}

function decodeBase64(blob: string, expected: number): Uint8Array {
  const native = (Uint8Array as Base64Uint8ArrayConstructor).fromBase64;
  if (native) {
    const result = native(blob);
    if (result.length !== expected) throw new Error(`Unexpected SAM2 tensor size: ${result.length}`);
    return result;
  }
  const binary = atob(blob);
  if (binary.length !== expected) throw new Error(`Unexpected SAM2 tensor size: ${binary.length}`);
  const result = new Uint8Array(expected);
  for (let index = 0; index < expected; index++) result[index] = binary.charCodeAt(index);
  return result;
}

function loadFeatures(key: string, blobs: [string, string, string] | null): Uint16Array[] {
  if (blobs) {
    const features = blobs.map((blob, index) => {
      const length = FEATURE_SHAPES[index].reduce((total, value) => total * value, 1);
      const bytes = decodeBase64(blob, length * 2);
      return new Uint16Array(bytes.buffer, bytes.byteOffset, length);
    });
    imageCache.delete(key);
    imageCache.set(key, features);
    if (imageCache.size > MAX_CACHED_IMAGES) imageCache.delete(imageCache.keys().next().value);
  }
  const features = imageCache.get(key);
  if (!features) throw new Error("SAM2 image features are not cached for this frame");
  imageCache.delete(key);
  imageCache.set(key, features);
  return features;
}

function axis(size: number): { lower: Uint16Array; upper: Uint16Array; weight: Float32Array } {
  const cached = resizeCache.get(size);
  if (cached) return cached;
  const lower = new Uint16Array(size);
  const upper = new Uint16Array(size);
  const weight = new Float32Array(size);
  for (let index = 0; index < size; index++) {
    const source = Math.max(0, Math.min(MASK_SIZE - 1, (index + 0.5) * MASK_SIZE / size - 0.5));
    lower[index] = Math.floor(source);
    upper[index] = Math.min(lower[index] + 1, MASK_SIZE - 1);
    weight[index] = source - lower[index];
  }
  const result = { lower, upper, weight };
  resizeCache.set(size, result);
  return result;
}

function resizeMask(logits: Float32Array, offset: number, width: number, height: number): SAM2OutputItem {
  const xAxis = axis(width);
  const yAxis = axis(height);
  const mask = new Uint8ClampedArray(width * height);
  let xtl = width; let ytl = height; let xbr = -1; let ybr = -1;
  for (let y = 0; y < height; y++) {
    const y0 = yAxis.lower[y]; const y1 = yAxis.upper[y]; const yw = yAxis.weight[y];
    for (let x = 0; x < width; x++) {
      const x0 = xAxis.lower[x]; const x1 = xAxis.upper[x]; const xw = xAxis.weight[x];
      const top = logits[offset + y0 * MASK_SIZE + x0] * (1 - xw) + logits[offset + y0 * MASK_SIZE + x1] * xw;
      const bottom = logits[offset + y1 * MASK_SIZE + x0] * (1 - xw) + logits[offset + y1 * MASK_SIZE + x1] * xw;
      if (top * (1 - yw) + bottom * yw > 0) {
        mask[y * width + x] = 1;
        if (x < xtl) xtl = x; if (y < ytl) ytl = y; if (x > xbr) xbr = x; if (y > ybr) ybr = y;
      }
    }
  }
  if (xbr < xtl) return { points: new Uint8ClampedArray(0), bounds: [0, 0, 0, 0] };
  const cropped = new Uint8ClampedArray((xbr - xtl + 1) * (ybr - ytl + 1));
  for (let y = ytl; y <= ybr; y++) cropped.set(mask.subarray(y * width + xtl, y * width + xbr + 1), (y - ytl) * (xbr - xtl + 1));
  return { points: cropped, bounds: [xtl, ytl, xbr, ybr] };
}

async function decode(body: DecodeBody): Promise<SAM2OutputItem> {
  if (!decoder) throw new Error("SAM2 worker was not initialized");
  if (body.coordinates.length !== PROMPT_SLOTS * 2 || body.labels.length !== PROMPT_SLOTS) {
    throw new Error("SAM2 prompt tensors do not match the decoder capacity");
  }
  const features = loadFeatures(body.imageKey, body.imageBlobs);
  const outputs = await decoder.run({
    high_res_0: new Tensor("float16", features[0], FEATURE_SHAPES[0]),
    high_res_1: new Tensor("float16", features[1], FEATURE_SHAPES[1]),
    image_embed: new Tensor("float16", features[2], FEATURE_SHAPES[2]),
    point_coords: new Tensor("float32", body.coordinates, [1, PROMPT_SLOTS, 2]),
    point_labels: new Tensor("int64", body.labels, [1, PROMPT_SLOTS]),
  });
  const masks = (await outputs.masks.getData()) as unknown as Float32Array;
  const scores = (await outputs.scores.getData()) as unknown as Float32Array;
  let best = 0;
  for (let index = 1; index < scores.length; index++) if (scores[index] > scores[best]) best = index;
  return resizeMask(masks, best * MASK_SIZE * MASK_SIZE, body.width, body.height);
}

// eslint-disable-next-line no-restricted-globals
if ((self as any).importScripts) {
  onmessage = async (event: MessageEvent<WorkerInput>) => {
    const { action, requestID, payload } = event.data;
    try {
      if (action === WorkerAction.INIT) {
        const init = payload as InitBody;
        if (!decoder) decoder = await createDecoder(init.decoderURL, init.executionProvider);
        postMessage({ action, requestID });
      } else {
        postMessage({ action, requestID, payload: await decode(payload as DecodeBody) });
      }
    } catch (error) {
      postMessage({ action, requestID, error: message(error) });
    }
  };
}
