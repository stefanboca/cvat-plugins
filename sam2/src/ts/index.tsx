import { LRUCache } from "@modules/lru-cache";
import { type InteractorResults, type Job, type MLModel, ShapeType, Source } from "cvat-core-wrapper";
import { APIWrapperEnterOptions, ComponentBuilder, PluginEntryPoint } from "components/plugins-entrypoint";
import { DecodeBody, ExecutionProvider, InitBody, SAM2OutputItem, WorkerAction } from "./inference.worker";

type ROI = [number, number, number, number];
interface Arguments {
  frame: number; pos_points: number[][]; neg_points: number[][]; obj_bbox: number[][]; roi?: ROI;
}
interface ServerResult { version: 1; dtype: "float16"; image_blobs: [string, string, string] }
interface Response<T> { requestID: number; payload?: T; error?: string }
interface Plugin {
  name: string; description: string;
  cvat: { jobs: { get: { leave: (plugin: Plugin, results: Job[], query: { jobID?: number }) => Promise<Job[]> } }; lambda: { call: {
    enter: (plugin: Plugin, taskID: number, model: MLModel, args: Arguments) => Promise<null | APIWrapperEnterOptions>;
    leave: (plugin: Plugin, result: unknown, taskID: number, model: MLModel, args: Arguments) => Promise<unknown>;
  } } };
  data: {
    initialized: boolean; initialization: Promise<void> | null; worker: Worker; jobs: Record<number, Job>;
    modelID: string; modelURL: string; images: LRUCache<string, true>; nextRequestID: number;
    requests: Map<number, { resolve: (payload: any) => void; reject: (error: Error) => void }>;
  };
  callbacks: { mask2Rle: ((points: Uint8ClampedArray) => number[]) | null };
}

function key(task: number, frame: number, roi?: ROI): string { return `${task}_${frame}_${roi?.join("_") ?? "full"}`; }
function send<T>(plugin: Plugin, action: WorkerAction, payload: InitBody | DecodeBody): Promise<T> {
  return new Promise((resolve, reject) => {
    const requestID = plugin.data.nextRequestID++;
    plugin.data.requests.set(requestID, { resolve, reject });
    plugin.data.worker.postMessage({ action, requestID, payload });
  });
}
function executionProvider(): ExecutionProvider {
  const value = window.localStorage.getItem("cvat.sam2.executionProvider");
  return value === "webgpu" || value === "wasm" ? value : "auto";
}
function debugEnabled(): boolean {
  return window.localStorage.getItem("cvat.sam2.debug") === "1";
}
function initialize(plugin: Plugin): Promise<void> {
  if (plugin.data.initialized) return Promise.resolve();
  if (!plugin.data.initialization) {
    plugin.data.initialization = send<void>(plugin, WorkerAction.INIT, {
      decoderURL: plugin.data.modelURL,
      executionProvider: executionProvider(),
    })
      .then(() => { plugin.data.initialized = true; })
      .catch((error) => { plugin.data.initialization = null; throw error; });
  }
  return plugin.data.initialization;
}
function normalize(point: number[], roi: ROI | undefined, width: number, height: number): [number, number] {
  return [(point[0] - (roi?.[0] ?? 0)) / width * 1024, (point[1] - (roi?.[1] ?? 0)) / height * 1024];
}
function makeBody(imageKey: string, blobs: [string, string, string] | null, args: Arguments, width: number, height: number): DecodeBody {
  if (args.obj_bbox.length !== 0 && args.obj_bbox.length !== 2) throw new Error("SAM2 requires two corners for one box");
  const count = args.pos_points.length + args.neg_points.length;
  if (!args.obj_bbox.length && !count) throw new Error("SAM2 requires a box or a point prompt");
  const boxSlots = args.obj_bbox.length ? 2 : 0;
  const coordinates = new Float32Array((boxSlots + count) * 2);
  const labels = new BigInt64Array(boxSlots + count);
  if (args.obj_bbox.length) {
    const first = normalize(args.obj_bbox[0], args.roi, width, height);
    const second = normalize(args.obj_bbox[1], args.roi, width, height);
    coordinates.set([Math.min(first[0], second[0]), Math.min(first[1], second[1]), Math.max(first[0], second[0]), Math.max(first[1], second[1])]);
    labels[0] = 2n; labels[1] = 3n;
  }
  const prompts = [...args.pos_points.map((point) => ({ point, label: 1n })), ...args.neg_points.map((point) => ({ point, label: 0n }))];
  prompts.forEach(({ point, label }, index) => { coordinates.set(normalize(point, args.roi, width, height), (index + boxSlots) * 2); labels[index + boxSlots] = label; });
  return { imageKey, imageBlobs: blobs, coordinates, labels, width, height, debug: debugEnabled() };
}
function parse(value: unknown): ServerResult {
  const result = value as Partial<ServerResult> | null;
  if (!result || result.version !== 1 || result.dtype !== "float16" || !Array.isArray(result.image_blobs) || result.image_blobs.length !== 3 || !result.image_blobs.every((blob) => typeof blob === "string")) {
    throw new Error("SAM2 server returned an unsupported feature payload");
  }
  return result as ServerResult;
}

const plugin: Plugin = {
  name: "Segment Anything 2",
  description: "Runs the SAM2 decoder in the browser",
  cvat: {
    jobs: { get: { async leave(_plugin, results, query) { if (typeof query.jobID === "number") [plugin.data.jobs[query.jobID]] = results; return results; } } },
    lambda: { call: {
      async enter(current, taskID, model, args) {
        if (model.id !== current.data.modelID) return null;
        await initialize(current);
        return current.data.images.get(key(taskID, args.frame, args.roi)) ? { preventMethodCall: true } : null;
      },
      async leave(current, result, taskID, model, args) {
        if (model.id !== current.data.modelID) return result;
        const job = Object.values(current.data.jobs).find((item) => item.taskId === taskID && args.frame >= item.startFrame && args.frame <= item.stopFrame);
        if (!job) throw new Error("Could not find a job corresponding to the SAM2 request");
        current.data.jobs = { [job.id]: job };
        const frame = await job.frames.get(args.frame);
        const width = args.roi ? args.roi[2] - args.roi[0] : frame.width;
        const height = args.roi ? args.roi[3] - args.roi[1] : frame.height;
        const imageKey = key(taskID, args.frame, args.roi);
        const server = result ? parse(result) : null;
        const body = makeBody(imageKey, server?.image_blobs ?? null, args, width, height);
        if (body.debug) {
          console.info("SAM2 browser request", {
            imageKey,
            imageFeaturesReceived: !!server,
            positivePoints: args.pos_points.map((point) => [...point]),
            negativePoints: args.neg_points.map((point) => [...point]),
            box: args.obj_bbox.map((point) => [...point]),
            decoderPrompts: Array.from(body.labels, (label, index) => label === -1n ? null : {
              label: Number(label),
              x: body.coordinates[index * 2],
              y: body.coordinates[index * 2 + 1],
            }).filter((prompt) => prompt !== null),
          });
        }
        const item = await send<SAM2OutputItem>(current, WorkerAction.DECODE, body);
        if (server) current.data.images.set(imageKey, true);
        let rle = current.callbacks.mask2Rle!(item.points);
        if (rle.length < 2) rle = [0, 0, 0, 0, 0];
        else {
          const bounds = [...item.bounds];
          if (args.roi) { bounds[0] += args.roi[0]; bounds[1] += args.roi[1]; bounds[2] += args.roi[0]; bounds[3] += args.roi[1]; }
          rle.push(...bounds);
        }
        return { shapes: [{ points: rle, group: 0, source: Source.SEMI_AUTO, occluded: false, rotation: 0, type: ShapeType.MASK, attributes: [] }] } as unknown as InteractorResults;
      },
    } },
  },
  data: {
    initialized: false, initialization: null, worker: new Worker(new URL("./inference.worker", import.meta.url)), jobs: {},
    modelID: "pth-facebookresearch-sam2", modelURL: "/assets/sam2_decoder.onnx", images: new LRUCache({ max: 32 }), nextRequestID: 0, requests: new Map(),
  },
  callbacks: { mask2Rle: null },
};

const builder: ComponentBuilder = ({ core }) => {
  plugin.callbacks.mask2Rle = core.utils.mask2Rle;
  plugin.data.worker.onmessage = (event: MessageEvent<Response<unknown>>) => {
    const request = plugin.data.requests.get(event.data.requestID);
    if (!request) return;
    plugin.data.requests.delete(event.data.requestID);
    if (event.data.error) request.reject(new Error(`SAM2 worker error: ${event.data.error}`)); else request.resolve(event.data.payload);
  };
  plugin.data.worker.onerror = (event) => { for (const request of plugin.data.requests.values()) request.reject(new Error(event.message)); plugin.data.requests.clear(); };
  core.plugins.register(plugin);
  return { name: plugin.name, destructor: () => { plugin.data.worker.terminate(); plugin.data.requests.clear(); plugin.data.images.clear(); } };
};
function register(): void {
  if (Object.prototype.hasOwnProperty.call(window, "cvatUI")) (window as any as { cvatUI: { registerComponent: PluginEntryPoint } }).cvatUI.registerComponent(builder);
}
window.addEventListener("plugins.ready", register, { once: true });
