const fs = require("fs");
const path = require("path");
const { createRequire } = require("module");
const root = process.env.CVAT_UI_ROOT;
if (!root) throw new Error("SAM2 type checking requires CVAT_UI_ROOT.");
const resolved = path.resolve(root);
const requireCVAT = createRequire(path.join(resolved, "package.json"));
const ts = requireCVAT("typescript");
const configPath = path.join(resolved, "tsconfig.json");
const config = ts.readConfigFile(configPath, ts.sys.readFile);
const modules = path.resolve(resolved, "..", "node_modules");
const source = path.resolve(__dirname, "..", "src", "ts");
const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, path.dirname(configPath), { paths: {
  "@modules/*": [path.join(modules, "*")],
  "@modules/onnxruntime-web/dist/ort.webgpu.min.mjs": [path.join(modules, "onnxruntime-common", "dist", "esm", "index.d.ts")],
  react: [path.join(modules, "react")],
} }, configPath);
const files = fs.readdirSync(source).filter((name) => /\.tsx?$/.test(name)).map((name) => path.join(source, name));
const diagnostics = [...parsed.errors, ...ts.getPreEmitDiagnostics(ts.createProgram([...new Set([...parsed.fileNames, ...files])], parsed.options))];
if (diagnostics.length) {
  process.stderr.write(ts.formatDiagnosticsWithColorAndContext(diagnostics, { getCanonicalFileName: (name) => name, getCurrentDirectory: ts.sys.getCurrentDirectory, getNewLine: () => ts.sys.newLine }));
  process.exitCode = 1;
} else process.stdout.write("SAM2 TypeScript check passed.\n");
