const fs = require("fs");
const path = require("path");
const { createRequire } = require("module");

const cvatUIRoot = process.env.CVAT_UI_ROOT;
if (!cvatUIRoot) {
  throw new Error("Server-side SAM3 type checking requires CVAT_UI_ROOT.");
}

const resolvedCVATUIRoot = path.resolve(cvatUIRoot);
const requireFromCVAT = createRequire(
  path.join(resolvedCVATUIRoot, "package.json"),
);
const ts = requireFromCVAT("typescript");
const configPath = path.join(resolvedCVATUIRoot, "tsconfig.json");
const configFile = ts.readConfigFile(configPath, ts.sys.readFile);

if (configFile.error) {
  process.stderr.write(
    ts.formatDiagnosticsWithColorAndContext([configFile.error], {
      getCanonicalFileName: (fileName) => fileName,
      getCurrentDirectory: ts.sys.getCurrentDirectory,
      getNewLine: () => ts.sys.newLine,
    }),
  );
  process.exitCode = 1;
} else {
  const nodeModules = path.resolve(resolvedCVATUIRoot, "..", "node_modules");
  const pluginSource = path.resolve(__dirname, "..", "src", "ts");
  const parsed = ts.parseJsonConfigFileContent(
    configFile.config,
    ts.sys,
    path.dirname(configPath),
    {
      paths: {
        "@modules/*": [path.join(nodeModules, "*")],
        react: [path.join(nodeModules, "react")],
      },
    },
    configPath,
  );
  const pluginFiles = fs
    .readdirSync(pluginSource)
    .filter((fileName) => /\.tsx?$/.test(fileName))
    .map((fileName) => path.join(pluginSource, fileName));
  const program = ts.createProgram(
    [...new Set([...parsed.fileNames, ...pluginFiles])],
    parsed.options,
  );
  const diagnostics = [...parsed.errors, ...ts.getPreEmitDiagnostics(program)];

  if (diagnostics.length) {
    process.stderr.write(
      ts.formatDiagnosticsWithColorAndContext(diagnostics, {
        getCanonicalFileName: (fileName) => fileName,
        getCurrentDirectory: ts.sys.getCurrentDirectory,
        getNewLine: () => ts.sys.newLine,
      }),
    );
    process.exitCode = 1;
  } else {
    process.stdout.write("Server-side SAM3 TypeScript check passed.\n");
  }
}
