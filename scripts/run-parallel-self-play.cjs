const { existsSync } = require("node:fs");
const { join } = require("node:path");
const { spawnSync } = require("node:child_process");

const root = process.cwd();
const mlDir = join(root, "ml");
const winPython = join(mlDir, ".venv", "Scripts", "python.exe");
const posixPython = join(mlDir, ".venv", "bin", "python");
const python = existsSync(winPython) ? winPython : existsSync(posixPython) ? posixPython : "python";

const args = ["-m", "oetongsu_ml.parallel_self_play", ...process.argv.slice(2)];
const result = spawnSync(python, args, {
  cwd: mlDir,
  stdio: "inherit",
  shell: false
});

process.exit(result.status ?? 1);
