const { existsSync } = require("node:fs");
const { join } = require("node:path");
const { spawnSync } = require("node:child_process");

const root = process.cwd();
const mlDir = join(root, "ml");
const winPython = join(mlDir, ".venv", "Scripts", "python.exe");
const posixPython = join(mlDir, ".venv", "bin", "python");
const python = existsSync(winPython) ? winPython : existsSync(posixPython) ? posixPython : "python";

const result = spawnSync(python, ["-m", "oetongsu_ml.training_server", ...process.argv.slice(2)], {
  cwd: mlDir,
  stdio: "inherit",
  shell: false
});

process.exit(result.status ?? 1);
