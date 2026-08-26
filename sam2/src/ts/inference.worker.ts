import { InferenceSession, Tensor, env } from "@modules/onnxruntime-web/dist/ort.webgpu.min.mjs";

const FEATURE_SHAPES = [[1, 32, 256, 256], [1, 64, 128, 128], [1, 256, 64, 64]] as const;
const MASK_SIZE = 256;
let decoder: InferenceSession | null = null;
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
export type ImageFeatures = [Uint16Array, Uint16Array, Uint16Array];
export interface DecodeBody {
  imageKey: string;
  imageFeatures: ImageFeatures;
  imageFeaturesReceived: boolean;
  coordinates: Float32Array;
  labels: BigInt64Array;
  width: number;
  height: number;
  debug: boolean;
}
export interface SAM2OutputItem { points: Uint8ClampedArray; bounds: [number, number, number, number] }
interface WorkerInput { action: WorkerAction; requestID: number; payload: InitBody | DecodeBody }

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
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
  if (!body.labels.length || body.coordinates.length !== body.labels.length * 2) {
    throw new Error("SAM2 prompt tensors do not match the decoder inputs");
  }
  const outputs = await decoder.run({
    high_res_0: new Tensor("float16", body.imageFeatures[0], FEATURE_SHAPES[0]),
    high_res_1: new Tensor("float16", body.imageFeatures[1], FEATURE_SHAPES[1]),
    image_embed: new Tensor("float16", body.imageFeatures[2], FEATURE_SHAPES[2]),
    point_coords: new Tensor("float32", body.coordinates, [1, body.labels.length, 2]),
    point_labels: new Tensor("int64", body.labels, [1, body.labels.length]),
  });
  const masks = (await outputs.masks.getData()) as unknown as Float32Array;
  const scores = (await outputs.scores.getData()) as unknown as Float32Array;
  let best = 0;
  for (let index = 1; index < scores.length; index++) if (scores[index] > scores[best]) best = index;
  const result = resizeMask(masks, best * MASK_SIZE * MASK_SIZE, body.width, body.height);
  if (body.debug) {
    console.info("SAM2 worker result", {
      imageKey: body.imageKey,
      imageFeaturesReceived: body.imageFeaturesReceived,
      prompts: Array.from(body.labels, (label, index) => label === -1n ? null : {
        label: Number(label),
        x: body.coordinates[index * 2],
        y: body.coordinates[index * 2 + 1],
      }).filter((prompt) => prompt !== null),
      scores: Array.from(scores),
      selectedMask: best,
      bounds: result.bounds,
    });
  }
  return result;
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
