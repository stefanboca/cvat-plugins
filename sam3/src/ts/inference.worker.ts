import {
  InferenceSession,
  Tensor,
  env,
} from "@modules/onnxruntime-web/webgpu";

const FPN_SHAPES = [
  [1, 256, 288, 288],
  [1, 256, 144, 144],
  [1, 256, 72, 72],
] as const;

const MAX_BOXES = 1;
const MAX_POINTS = 64;
const MASK_SIZE = 288;
const TEXT_TOKENS = 32;
const TEXT_CHANNELS = 256;
const MAX_CACHED_IMAGES = 2;
const MAX_CACHED_TEXTS = 32;

let decoder: InferenceSession | null = null;
const imageFeatureCache = new Map<string, Uint16Array[]>();
const textFeatureCache = new Map<
  string,
  { features: Uint16Array; mask: Uint8Array }
>();
const resizeAxisCache = new Map<
  number,
  { lower: Uint16Array; upper: Uint16Array; weight: Float32Array }
>();

env.wasm.wasmPaths = "/assets/";

export enum WorkerAction {
  INIT = "init",
  DECODE = "decode",
}

export interface InitBody {
  decoderURL: string;
}

export interface DecodeBody {
  imageKey: string;
  imageBlobs: [string, string, string] | null;
  textKey: string;
  textFeatureBlob: string | null;
  textMaskBlob: string | null;
  boxes: Float32Array;
  boxValid: Uint8Array;
  boxLabels: BigInt64Array;
  points: Float32Array;
  pointValid: Uint8Array;
  pointLabels: BigInt64Array;
  width: number;
  height: number;
}

export interface WorkerInput {
  action: WorkerAction;
  requestID: number;
  payload: InitBody | DecodeBody;
}

export interface SAM3OutputItem {
  points: Uint8ClampedArray;
  bounds: [number, number, number, number];
}

function errorToMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }

  console.error(error);
  return "Unknown error, please check console";
}

interface Uint8ArrayConstructorWithBase64 extends Uint8ArrayConstructor {
  fromBase64?: (value: string) => Uint8Array;
}

function decodeBase64(blob: string, expectedLength: number): Uint8Array {
  const nativeDecoder = (Uint8Array as Uint8ArrayConstructorWithBase64)
    .fromBase64;
  if (nativeDecoder) {
    const result = nativeDecoder(blob);
    if (result.length !== expectedLength) {
      throw new Error(
        `Unexpected SAM3 tensor size: got ${result.length} bytes, expected ${expectedLength}`,
      );
    }
    return result;
  }

  const binary = atob(blob);
  if (binary.length !== expectedLength) {
    throw new Error(
      `Unexpected SAM3 tensor size: got ${binary.length} bytes, expected ${expectedLength}`,
    );
  }
  const result = new Uint8Array(expectedLength);
  for (let index = 0; index < expectedLength; index++) {
    result[index] = binary.charCodeAt(index);
  }
  return result;
}

function decodeFloat16(blob: string, expectedLength: number): Uint16Array {
  const bytes = decodeBase64(blob, expectedLength * 2);
  return new Uint16Array(bytes.buffer, bytes.byteOffset, expectedLength);
}

function putLRU<T>(
  cache: Map<string, T>,
  key: string,
  value: T,
  max: number,
): void {
  cache.delete(key);
  cache.set(key, value);
  if (cache.size > max) {
    cache.delete(cache.keys().next().value);
  }
}

function getLRU<T>(cache: Map<string, T>, key: string): T | null {
  const value = cache.get(key);
  if (!value) {
    return null;
  }
  cache.delete(key);
  cache.set(key, value);
  return value;
}

function loadImageFeatures(
  imageKey: string,
  blobs: [string, string, string] | null,
): Uint16Array[] {
  if (blobs) {
    const features = blobs.map((blob, index) => {
      const shape = FPN_SHAPES[index];
      const length = shape.reduce(
        (accumulator, dimension) => accumulator * dimension,
        1,
      );
      return decodeFloat16(blob, length);
    });
    putLRU(imageFeatureCache, imageKey, features, MAX_CACHED_IMAGES);
  }

  const features = getLRU(imageFeatureCache, imageKey);
  if (!features) {
    throw new Error("SAM3 image features are not cached for this frame");
  }

  return features;
}

function decodeBytes(blob: string, expectedLength: number): Uint8Array {
  return decodeBase64(blob, expectedLength);
}

function loadTextFeatures(
  textKey: string,
  featureBlob: string | null,
  maskBlob: string | null,
): { features: Uint16Array; mask: Uint8Array } {
  if ((featureBlob === null) !== (maskBlob === null)) {
    throw new Error(
      "SAM3 text feature and mask blobs must be supplied together",
    );
  }
  if (featureBlob && maskBlob) {
    putLRU(
      textFeatureCache,
      textKey,
      {
        features: decodeFloat16(featureBlob, TEXT_TOKENS * TEXT_CHANNELS),
        mask: decodeBytes(maskBlob, TEXT_TOKENS),
      },
      MAX_CACHED_TEXTS,
    );
  }

  const text = getLRU(textFeatureCache, textKey);
  if (!text) {
    throw new Error("SAM3 text features are not cached for this prompt");
  }
  return text;
}

function getResizeAxis(outputSize: number): {
  lower: Uint16Array;
  upper: Uint16Array;
  weight: Float32Array;
} {
  const cached = resizeAxisCache.get(outputSize);
  if (cached) {
    return cached;
  }

  const lower = new Uint16Array(outputSize);
  const upper = new Uint16Array(outputSize);
  const weight = new Float32Array(outputSize);
  for (let index = 0; index < outputSize; index++) {
    const source = Math.max(
      0,
      Math.min(
        MASK_SIZE - 1,
        (index + 0.5) * (MASK_SIZE / outputSize) - 0.5,
      ),
    );
    lower[index] = Math.floor(source);
    upper[index] = Math.min(lower[index] + 1, MASK_SIZE - 1);
    weight[index] = source - lower[index];
  }
  const axis = { lower, upper, weight };
  resizeAxisCache.set(outputSize, axis);
  return axis;
}

function resizeAndCropMask(
  logits: Float32Array,
  width: number,
  height: number,
): SAM3OutputItem {
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1
  ) {
    throw new Error(`Invalid SAM3 output size: ${width}x${height}`);
  }

  const mask = new Uint8ClampedArray(width * height);
  let xtl = width;
  let ytl = height;
  let xbr = -1;
  let ybr = -1;
  const xAxis = getResizeAxis(width);
  const yAxis = getResizeAxis(height);

  for (let y = 0; y < height; y++) {
    const y0 = yAxis.lower[y];
    const y1 = yAxis.upper[y];
    const yWeight = yAxis.weight[y];
    const yInverseWeight = 1 - yWeight;
    const topOffset = y0 * MASK_SIZE;
    const bottomOffset = y1 * MASK_SIZE;
    const outputOffset = y * width;

    for (let x = 0; x < width; x++) {
      const x0 = xAxis.lower[x];
      const x1 = xAxis.upper[x];
      const xWeight = xAxis.weight[x];
      const xInverseWeight = 1 - xWeight;
      const top =
        logits[topOffset + x0] * xInverseWeight +
        logits[topOffset + x1] * xWeight;
      const bottom =
        logits[bottomOffset + x0] * xInverseWeight +
        logits[bottomOffset + x1] * xWeight;

      if (top * yInverseWeight + bottom * yWeight > 0) {
        mask[outputOffset + x] = 1;
        if (x < xtl) xtl = x;
        if (y < ytl) ytl = y;
        if (x > xbr) xbr = x;
        if (y > ybr) ybr = y;
      }
    }
  }

  if (xbr < xtl || ybr < ytl) {
    return {
      points: new Uint8ClampedArray(0),
      bounds: [0, 0, 0, 0],
    };
  }

  const croppedWidth = xbr - xtl + 1;
  const cropped = new Uint8ClampedArray(croppedWidth * (ybr - ytl + 1));
  for (let y = ytl; y <= ybr; y++) {
    cropped.set(
      mask.subarray(y * width + xtl, y * width + xbr + 1),
      (y - ytl) * croppedWidth,
    );
  }

  return { points: cropped, bounds: [xtl, ytl, xbr, ybr] };
}

async function decode(body: DecodeBody): Promise<SAM3OutputItem> {
  if (!decoder) {
    throw new Error("SAM3 worker was not initialized");
  }
  if (
    body.boxes.length !== MAX_BOXES * 4 ||
    body.points.length !== MAX_POINTS * 2
  ) {
    throw new Error(
      "SAM3 prompt tensors do not match the exported decoder capacities",
    );
  }

  const features = loadImageFeatures(body.imageKey, body.imageBlobs);
  const text = loadTextFeatures(
    body.textKey,
    body.textFeatureBlob,
    body.textMaskBlob,
  );
  const inputs: Record<string, Tensor> = {
    fpn_288: new Tensor("float16", features[0], FPN_SHAPES[0]),
    fpn_144: new Tensor("float16", features[1], FPN_SHAPES[1]),
    fpn_72: new Tensor("float16", features[2], FPN_SHAPES[2]),
    boxes: new Tensor("float32", body.boxes, [1, MAX_BOXES, 4]),
    box_valid: new Tensor("bool", body.boxValid, [1, MAX_BOXES]),
    box_labels: new Tensor("int64", body.boxLabels, [1, MAX_BOXES]),
    points: new Tensor("float32", body.points, [1, MAX_POINTS, 2]),
    point_valid: new Tensor("bool", body.pointValid, [1, MAX_POINTS]),
    point_labels: new Tensor("int64", body.pointLabels, [1, MAX_POINTS]),
    language_features: new Tensor("float16", text.features, [
      TEXT_TOKENS,
      1,
      TEXT_CHANNELS,
    ]),
    language_mask: new Tensor("bool", text.mask, [1, TEXT_TOKENS]),
  };
  const outputs = await decoder.run(inputs);
  const logits =
    (await outputs.mask_logits.getData()) as unknown as Float32Array;
  return resizeAndCropMask(logits, body.width, body.height);
}

// eslint-disable-next-line no-restricted-globals
if ((self as any).importScripts) {
  onmessage = async (event: MessageEvent<WorkerInput>) => {
    const { action, payload, requestID } = event.data;
    try {
      if (action === WorkerAction.INIT) {
        if (!decoder) {
          decoder = await InferenceSession.create(
            (payload as InitBody).decoderURL,
            { executionProviders: ["webgpu", "wasm"] },
          );
        }
        postMessage({ action, requestID });
      } else if (action === WorkerAction.DECODE) {
        const item = await decode(payload as DecodeBody);
        postMessage({ action, requestID, payload: item });
      }
    } catch (error) {
      postMessage({ action, requestID, error: errorToMessage(error) });
    }
  };
}
