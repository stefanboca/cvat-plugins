import { InferenceSession, Tensor, env } from "@modules/onnxruntime-web";

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
const imageFeatureCache = new Map<string, Float32Array[]>();
const textFeatureCache = new Map<
  string,
  { features: Float32Array; mask: Uint8Array }
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

function decodeFloat16(blob: string, expectedLength: number): Float32Array {
  const binary = atob(blob);
  if (binary.length !== expectedLength * 2) {
    throw new Error(
      `Unexpected SAM3 feature size: got ${binary.length} bytes, expected ${expectedLength * 2}`,
    );
  }

  const result = new Float32Array(expectedLength);
  /* eslint-disable no-bitwise */
  for (let index = 0; index < expectedLength; index++) {
    const value =
      binary.charCodeAt(index * 2) | (binary.charCodeAt(index * 2 + 1) << 8);
    const sign = value & 0x8000 ? -1 : 1;
    const exponent = (value >> 10) & 0x1f;
    const fraction = value & 0x03ff;

    if (exponent === 0) {
      result[index] = sign * 2 ** -14 * (fraction / 1024);
    } else if (exponent === 0x1f) {
      result[index] = fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
    } else {
      result[index] = sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
    }
  }
  /* eslint-enable no-bitwise */

  return result;
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
): Float32Array[] {
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
  const binary = atob(blob);
  if (binary.length !== expectedLength) {
    throw new Error(
      `Unexpected SAM3 byte tensor size: got ${binary.length}, expected ${expectedLength}`,
    );
  }
  const result = new Uint8Array(expectedLength);
  for (let index = 0; index < expectedLength; index++) {
    result[index] = binary.charCodeAt(index);
  }
  return result;
}

function loadTextFeatures(
  textKey: string,
  featureBlob: string | null,
  maskBlob: string | null,
): { features: Float32Array; mask: Uint8Array } {
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

  for (let y = 0; y < height; y++) {
    const sourceY = Math.max(
      0,
      Math.min(MASK_SIZE - 1, (y + 0.5) * (MASK_SIZE / height) - 0.5),
    );
    const y0 = Math.floor(sourceY);
    const y1 = Math.min(y0 + 1, MASK_SIZE - 1);
    const yWeight = sourceY - y0;

    for (let x = 0; x < width; x++) {
      const sourceX = Math.max(
        0,
        Math.min(MASK_SIZE - 1, (x + 0.5) * (MASK_SIZE / width) - 0.5),
      );
      const x0 = Math.floor(sourceX);
      const x1 = Math.min(x0 + 1, MASK_SIZE - 1);
      const xWeight = sourceX - x0;
      const top =
        logits[y0 * MASK_SIZE + x0] * (1 - xWeight) +
        logits[y0 * MASK_SIZE + x1] * xWeight;
      const bottom =
        logits[y1 * MASK_SIZE + x0] * (1 - xWeight) +
        logits[y1 * MASK_SIZE + x1] * xWeight;

      if (top * (1 - yWeight) + bottom * yWeight > 0) {
        mask[y * width + x] = 1;
        xtl = Math.min(xtl, x);
        ytl = Math.min(ytl, y);
        xbr = Math.max(xbr, x);
        ybr = Math.max(ybr, y);
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
    fpn_288: new Tensor("float32", features[0], FPN_SHAPES[0]),
    fpn_144: new Tensor("float32", features[1], FPN_SHAPES[1]),
    fpn_72: new Tensor("float32", features[2], FPN_SHAPES[2]),
    boxes: new Tensor("float32", body.boxes, [1, MAX_BOXES, 4]),
    box_valid: new Tensor("bool", body.boxValid, [1, MAX_BOXES]),
    box_labels: new Tensor("int64", body.boxLabels, [1, MAX_BOXES]),
    points: new Tensor("float32", body.points, [1, MAX_POINTS, 2]),
    point_valid: new Tensor("bool", body.pointValid, [1, MAX_POINTS]),
    point_labels: new Tensor("int64", body.pointLabels, [1, MAX_POINTS]),
    language_features: new Tensor("float32", text.features, [
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
