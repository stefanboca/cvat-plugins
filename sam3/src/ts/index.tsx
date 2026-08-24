import React, { useState } from "react";
import Input from "@modules/antd/lib/input";
import { LRUCache } from "@modules/lru-cache";
import {
  type InteractorResults,
  type Job,
  type MLModel,
  ShapeType,
  Source,
} from "cvat-core-wrapper";
import {
  APIWrapperEnterOptions,
  ComponentBuilder,
  PluginEntryPoint,
} from "components/plugins-entrypoint";
import {
  DecodeBody,
  InitBody,
  SAM3OutputItem,
  WorkerAction,
} from "./inference.worker";

const MAX_BOXES = 1;
const MAX_POINTS = 64;

type ROI = [number, number, number, number];

interface SAM3ServerResult {
  version: 1;
  dtype: "float16";
  image_blobs: [string, string, string] | null;
  text: null | {
    prompt: string;
    feature_blob: string;
    mask_blob: string;
  };
}

interface WorkerResponse<T> {
  action: WorkerAction;
  requestID: number;
  payload?: T;
  error?: string;
}

interface SAM3Plugin {
  name: string;
  description: string;
  cvat: {
    lambda: {
      call: {
        enter: (
          plugin: SAM3Plugin,
          taskID: number,
          model: MLModel,
          args: InteractorArguments,
        ) => Promise<null | APIWrapperEnterOptions>;
        leave: (
          plugin: SAM3Plugin,
          result: unknown,
          taskID: number,
          model: MLModel,
          args: InteractorArguments,
        ) => Promise<unknown>;
      };
    };
    jobs: {
      get: {
        leave: (
          plugin: SAM3Plugin,
          results: Job[],
          query: { jobID?: number },
        ) => Promise<Job[]>;
      };
    };
  };
  data: {
    initialized: boolean;
    initialization: Promise<void> | null;
    worker: Worker;
    jobs: Record<number, Job>;
    modelID: string;
    modelURL: string;
    textPrompt: string;
    imageEmbeddings: LRUCache<string, true>;
    textEmbeddings: LRUCache<string, true>;
    nextRequestID: number;
    requests: Map<
      number,
      {
        resolve: (payload: any) => void;
        reject: (error: Error) => void;
      }
    >;
  };
  callbacks: {
    mask2Rle: ((points: Uint8ClampedArray) => number[]) | null;
  };
}

interface InteractorArguments {
  frame: number;
  pos_points: number[][];
  neg_points: number[][];
  obj_bbox: number[][];
  roi?: ROI;
  text_prompts?: string[];
}

function roiSignature(roi?: ROI): string {
  return roi ? roi.join("_") : "full";
}

function imageKey(taskID: number, frame: number, roi?: ROI): string {
  return `${taskID}_${frame}_${roiSignature(roi)}`;
}

function normalizedTextPrompt(plugin: SAM3Plugin): string {
  return plugin.data.textPrompt.trim() || "visual";
}

function sendWorkerRequest<T>(
  plugin: SAM3Plugin,
  action: WorkerAction,
  payload: InitBody | DecodeBody,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const requestID = plugin.data.nextRequestID++;
    plugin.data.requests.set(requestID, { resolve, reject });
    plugin.data.worker.postMessage({ action, requestID, payload });
  });
}

function initialize(plugin: SAM3Plugin): Promise<void> {
  if (plugin.data.initialized) {
    return Promise.resolve();
  }
  if (!plugin.data.initialization) {
    plugin.data.initialization = sendWorkerRequest<void>(
      plugin,
      WorkerAction.INIT,
      {
        decoderURL: plugin.data.modelURL,
      },
    )
      .then(() => {
        plugin.data.initialized = true;
      })
      .catch((error) => {
        plugin.data.initialization = null;
        throw error;
      });
  }

  return plugin.data.initialization;
}

function normalizePoint(
  point: number[],
  roi: ROI | undefined,
  width: number,
  height: number,
): [number, number] {
  const x = point[0] - (roi?.[0] ?? 0);
  const y = point[1] - (roi?.[1] ?? 0);
  return [x / width, y / height];
}

function makeDecodeBody(
  imageCacheKey: string,
  imageBlobs: [string, string, string] | null,
  textKey: string,
  textFeatureBlob: string | null,
  textMaskBlob: string | null,
  args: InteractorArguments,
  width: number,
  height: number,
): DecodeBody {
  if (args.obj_bbox.length !== 0 && args.obj_bbox.length !== 2) {
    throw new Error(
      `SAM3 expects one box represented by two corners, got ${args.obj_bbox.length} points`,
    );
  }
  const promptCount = args.pos_points.length + args.neg_points.length;
  if (promptCount > MAX_POINTS) {
    throw new Error(
      `SAM3 supports at most ${MAX_POINTS} point prompts, got ${promptCount}`,
    );
  }

  const boxes = new Float32Array(MAX_BOXES * 4);
  const boxValid = new Uint8Array(MAX_BOXES);
  const boxLabels = new BigInt64Array(MAX_BOXES);
  if (args.obj_bbox.length === 2) {
    const [first, second] = args.obj_bbox.map((point) =>
      normalizePoint(point, args.roi, width, height),
    );
    const xtl = Math.min(first[0], second[0]);
    const ytl = Math.min(first[1], second[1]);
    const xbr = Math.max(first[0], second[0]);
    const ybr = Math.max(first[1], second[1]);
    boxes.set([(xtl + xbr) / 2, (ytl + ybr) / 2, xbr - xtl, ybr - ytl]);
    boxValid[0] = 1;
    boxLabels[0] = 1n;
  }

  const points = new Float32Array(MAX_POINTS * 2);
  const pointValid = new Uint8Array(MAX_POINTS);
  const pointLabels = new BigInt64Array(MAX_POINTS);
  const prompts = [
    ...args.pos_points.map((point) => ({ point, label: 1n })),
    ...args.neg_points.map((point) => ({ point, label: 0n })),
  ];
  prompts.forEach(({ point, label }, index) => {
    points.set(normalizePoint(point, args.roi, width, height), index * 2);
    pointValid[index] = 1;
    pointLabels[index] = label;
  });

  return {
    imageKey: imageCacheKey,
    imageBlobs,
    textKey,
    textFeatureBlob,
    textMaskBlob,
    boxes,
    boxValid,
    boxLabels,
    points,
    pointValid,
    pointLabels,
    width,
    height,
  };
}

function parseServerResult(result: unknown): SAM3ServerResult {
  const value = result as Partial<SAM3ServerResult> | null;
  if (
    !value ||
    value.version !== 1 ||
    value.dtype !== "float16" ||
    (value.image_blobs !== null &&
      (!Array.isArray(value.image_blobs) ||
        value.image_blobs.length !== 3 ||
        !value.image_blobs.every((blob) => typeof blob === "string"))) ||
    (value.text !== null &&
      (!value.text ||
        typeof value.text.prompt !== "string" ||
        typeof value.text.feature_blob !== "string" ||
        typeof value.text.mask_blob !== "string"))
  ) {
    throw new Error("SAM3 server returned an unsupported feature payload");
  }

  return value as SAM3ServerResult;
}

const sam3Plugin: SAM3Plugin = {
  name: "Segment Anything 3",
  description: "Runs the SAM3 grounding decoder in the browser",
  cvat: {
    jobs: {
      get: {
        async leave(plugin, results, query): Promise<Job[]> {
          if (typeof query.jobID === "number") {
            [plugin.data.jobs[query.jobID]] = results;
          }
          return results;
        },
      },
    },
    lambda: {
      call: {
        async enter(
          plugin,
          taskID,
          model,
          args,
        ): Promise<null | APIWrapperEnterOptions> {
          if (model.id !== plugin.data.modelID) {
            return null;
          }

          await initialize(plugin);
          const textKey = normalizedTextPrompt(plugin);
          const hasImage = !!plugin.data.imageEmbeddings.get(
            imageKey(taskID, args.frame, args.roi),
          );
          const hasText = !!plugin.data.textEmbeddings.get(textKey);
          Object.assign(args, {
            text_prompts: [
              textKey,
              `cvat-sam3:${hasImage ? "" : "image"}:${hasText ? "" : "text"}`,
            ],
          });
          return hasImage && hasText ? { preventMethodCall: true } : null;
        },
        async leave(plugin, result, taskID, model, args): Promise<unknown> {
          if (model.id !== plugin.data.modelID) {
            return result;
          }

          const job = Object.values(plugin.data.jobs).find(
            (_job) =>
              _job.taskId === taskID &&
              args.frame >= _job.startFrame &&
              args.frame <= _job.stopFrame,
          );
          if (!job) {
            throw new Error(
              "Could not find a job corresponding to the SAM3 request",
            );
          }
          plugin.data.jobs = { [job.id]: job };

          const { height: imageHeight, width: imageWidth } =
            await job.frames.get(args.frame);
          const width = args.roi ? args.roi[2] - args.roi[0] : imageWidth;
          const height = args.roi ? args.roi[3] - args.roi[1] : imageHeight;
          const currentImageKey = imageKey(taskID, args.frame, args.roi);
          const currentTextKey = args.text_prompts?.[0] ?? "visual";
          const serverResult = result ? parseServerResult(result) : null;
          if (
            serverResult?.text &&
            serverResult.text.prompt !== currentTextKey
          ) {
            throw new Error(
              "SAM3 server returned text features for a different prompt",
            );
          }
          const item = await sendWorkerRequest<SAM3OutputItem>(
            plugin,
            WorkerAction.DECODE,
            makeDecodeBody(
              currentImageKey,
              serverResult?.image_blobs ?? null,
              currentTextKey,
              serverResult?.text?.feature_blob ?? null,
              serverResult?.text?.mask_blob ?? null,
              args,
              width,
              height,
            ),
          );
          if (serverResult?.image_blobs) {
            plugin.data.imageEmbeddings.set(currentImageKey, true);
          }
          if (serverResult?.text) {
            plugin.data.textEmbeddings.set(currentTextKey, true);
          }
          let rle = plugin.callbacks.mask2Rle!(item.points);
          if (rle.length < 2) {
            rle = [0, 0, 0, 0, 0];
          } else {
            const bounds = [...item.bounds] as [number, number, number, number];
            if (args.roi) {
              bounds[0] += args.roi[0];
              bounds[1] += args.roi[1];
              bounds[2] += args.roi[0];
              bounds[3] += args.roi[1];
            }
            rle.push(...bounds);
          }

          return {
            shapes: [
              {
                points: rle,
                group: 0,
                source: Source.SEMI_AUTO,
                occluded: false,
                rotation: 0,
                type: ShapeType.MASK,
                attributes: [],
              },
            ],
          } as unknown as InteractorResults;
        },
      },
    },
  },
  data: {
    initialized: false,
    initialization: null,
    worker: new Worker(new URL("./inference.worker", import.meta.url)),
    jobs: {},
    modelID: "pth-facebookresearch-sam3",
    modelURL: "/assets/sam3_decoder.onnx",
    textPrompt: "",
    imageEmbeddings: new LRUCache({ max: 32 }),
    textEmbeddings: new LRUCache({ max: 32 }),
    nextRequestID: 0,
    requests: new Map(),
  },
  callbacks: {
    mask2Rle: null,
  },
};

function TextPromptInput(): JSX.Element {
  const [value, setValue] = useState(sam3Plugin.data.textPrompt);
  return (
    <Input
      allowClear
      value={value}
      placeholder="Optional SAM3 text prompt"
      onKeyDown={(event) => event.stopPropagation()}
      onChange={(event) => {
        setValue(event.target.value);
        sam3Plugin.data.textPrompt = event.target.value;
      }}
    />
  );
}

const builder: ComponentBuilder = ({ core, dispatch, actionCreators }) => {
  sam3Plugin.callbacks.mask2Rle = core.utils.mask2Rle;
  sam3Plugin.data.worker.onmessage = (
    event: MessageEvent<WorkerResponse<unknown>>,
  ) => {
    const request = sam3Plugin.data.requests.get(event.data.requestID);
    if (!request) {
      return;
    }
    sam3Plugin.data.requests.delete(event.data.requestID);
    if (event.data.error) {
      request.reject(new Error(`SAM3 worker error: ${event.data.error}`));
    } else {
      request.resolve(event.data.payload);
    }
  };
  sam3Plugin.data.worker.onerror = (event: ErrorEvent) => {
    for (const request of sam3Plugin.data.requests.values()) {
      request.reject(
        event.error instanceof Error ? event.error : new Error(event.message),
      );
    }
    sam3Plugin.data.requests.clear();
  };
  core.plugins.register(sam3Plugin);
  dispatch(
    actionCreators.addUIComponent(
      "aiTools.interactors.extras",
      TextPromptInput,
      {
        weight: 10,
        shouldBeRendered: (_props, state: any) =>
          state.activeInteractor?.id === sam3Plugin.data.modelID,
      },
    ),
  );

  return {
    name: sam3Plugin.name,
    destructor: () => {
      sam3Plugin.data.worker.terminate();
      dispatch(
        actionCreators.removeUIComponent(
          "aiTools.interactors.extras",
          TextPromptInput,
        ),
      );
      sam3Plugin.data.requests.clear();
      sam3Plugin.data.jobs = {};
      sam3Plugin.data.imageEmbeddings.clear();
      sam3Plugin.data.textEmbeddings.clear();
      sam3Plugin.data.initialized = false;
      sam3Plugin.data.initialization = null;
    },
  };
};

function register(): void {
  if (Object.prototype.hasOwnProperty.call(window, "cvatUI")) {
    (
      window as any as { cvatUI: { registerComponent: PluginEntryPoint } }
    ).cvatUI.registerComponent(builder);
  }
}

window.addEventListener("plugins.ready", register, { once: true });
