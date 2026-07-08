param(
  [string]$ArtifactRoot = "D:\OetongsuArtifacts\a3_ablation_extract",
  [string]$RepoRoot = "D:\kim_dev\Checkmate",
  [int]$SeedLimit = 256,
  [int]$CheapGames = 4,
  [int]$CheapSimulations = 8,
  [int]$CheapMaxPlies = 80,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$mlRoot = Join-Path $RepoRoot "ml"
$python = Join-Path $mlRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
  $python = "python"
}

$required = @(
  "$ArtifactRoot\data\models\checkpoints\supervised_v0001.pt",
  "$ArtifactRoot\data\models\checkpoints\az_iter_000003.pt",
  "$ArtifactRoot\data\training\ablation_a3\ablation_a3_lr_0_001.pt",
  "$ArtifactRoot\data\selfplay\az_iter_000003.jsonl"
)

foreach ($path in $required) {
  if (-not (Test-Path -LiteralPath $path)) {
    throw "Missing required file: $path"
  }
}

$commands = @(
  @{
    Name = "seed_sensitivity_probe"
    Args = @(
      "-m", "oetongsu_ml.seed_sensitivity_probe",
      "--data", "$ArtifactRoot\data\selfplay\az_iter_000003.jsonl",
      "--resume", "$ArtifactRoot\data\models\checkpoints\supervised_v0001.pt",
      "--outputDir", "$ArtifactRoot\data\training\seed_probe",
      "--channels", "64",
      "--batchSize", "64",
      "--epochs", "1",
      "--lr", "0.001",
      "--seeds", "4", "7",
      "--limit", "$SeedLimit",
      "--noArena"
    )
  },
  @{
    Name = "cheap_validation_az_iter_000003"
    Args = @(
      "-m", "oetongsu_ml.cheap_validation_gate",
      "--candidate", "$ArtifactRoot\data\models\checkpoints\az_iter_000003.pt",
      "--champion", "$ArtifactRoot\data\models\checkpoints\supervised_v0001.pt",
      "--games", "$CheapGames",
      "--simulations", "$CheapSimulations",
      "--maxPlies", "$CheapMaxPlies",
      "--adjudicationDrawMargin", "1.5",
      "--output", "$ArtifactRoot\data\training\cheap_validation_az_iter_000003.json"
    )
  },
  @{
    Name = "cheap_validation_ablation_lr_0_001"
    Args = @(
      "-m", "oetongsu_ml.cheap_validation_gate",
      "--candidate", "$ArtifactRoot\data\training\ablation_a3\ablation_a3_lr_0_001.pt",
      "--champion", "$ArtifactRoot\data\models\checkpoints\supervised_v0001.pt",
      "--games", "$CheapGames",
      "--simulations", "$CheapSimulations",
      "--maxPlies", "$CheapMaxPlies",
      "--adjudicationDrawMargin", "1.5",
      "--output", "$ArtifactRoot\data\training\cheap_validation_ablation_lr_0_001.json"
    )
  }
)

Push-Location $mlRoot
try {
  foreach ($command in $commands) {
    Write-Host "==> $($command.Name)"
    Write-Host "$python $($command.Args -join ' ')"
    if (-not $DryRun) {
      & $python @($command.Args)
      if ($LASTEXITCODE -ne 0) {
        throw "$($command.Name) failed with exit code $LASTEXITCODE"
      }
    }
  }
}
finally {
  Pop-Location
}

Write-Host ""
Write-Host "A3 local validation outputs:"
Write-Host "- $ArtifactRoot\data\training\seed_probe\seed_sensitivity_summary.json"
Write-Host "- $ArtifactRoot\data\training\cheap_validation_az_iter_000003.json"
Write-Host "- $ArtifactRoot\data\training\cheap_validation_ablation_lr_0_001.json"
