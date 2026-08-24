const fs = require("fs");
const path = require("path");

const source = path.join(__dirname, "assets", "sam3_decoder.onnx");
const cvatUIRoot = process.env.CVAT_UI_ROOT;
if (!cvatUIRoot) {
  throw new Error("SAM3 requires CVAT_UI_ROOT at build time.");
}

const destination = path.join(
  path.resolve(cvatUIRoot),
  "dist",
  "assets",
  path.basename(source),
);
fs.mkdirSync(path.dirname(destination), { recursive: true });

const sourceStat = fs.statSync(source);
const destinationStat = fs.existsSync(destination)
  ? fs.statSync(destination)
  : null;
if (
  !destinationStat ||
  destinationStat.size !== sourceStat.size ||
  destinationStat.mtimeMs < sourceStat.mtimeMs
) {
  fs.copyFileSync(source, destination);
}
