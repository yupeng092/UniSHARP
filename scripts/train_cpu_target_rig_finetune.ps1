<#
Low-resolution Windows CPU smoke test for target-rig-conditioned fine-tuning.
It requires calibrated RE10K pairs and validates only the training chain; use
the NPU launcher for a full-resolution fine-tuning run.
#>
[CmdletBinding()]
param(
    [string]$DatasetManifestDir = $env:DATASET_MANIFEST_DIR,
    [string]$DataRootRe10K = $env:DATA_ROOT_RE10K,
    [string]$InitCheckpoint = $env:INIT_CHECKPOINT,
    [string]$OutRoot = "",
    [string]$RunName = "",
    [int]$Steps = 10,
    [int]$Threads = 8,
    [int]$NumWorkers = 2,
    [int]$Height = 256,
    [int]$Width = 384,
    [int]$TargetRigEmbedDim = 128,
    [double]$TargetRigTranslationScale = 1.0,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($DatasetManifestDir)) { throw "Set -DatasetManifestDir or DATASET_MANIFEST_DIR." }
if ([string]::IsNullOrWhiteSpace($DataRootRe10K)) { throw "Set -DataRootRe10K or DATA_ROOT_RE10K." }
if ([string]::IsNullOrWhiteSpace($InitCheckpoint)) { throw "Set -InitCheckpoint or INIT_CHECKPOINT." }
if (-not (Test-Path -LiteralPath $InitCheckpoint -PathType Leaf)) { throw "Checkpoint was not found: $InitCheckpoint" }
if ([string]::IsNullOrWhiteSpace($OutRoot)) { $OutRoot = Join-Path $repoRoot "outputs_cpu" }
if ([string]::IsNullOrWhiteSpace($RunName)) { $RunName = "target_rig_cpu_smoketest_$(Get-Date -Format 'yyyyMMdd_HHmmss')" }

$env:PYTHONPATH = "$repoRoot;$repoRoot\UniK3D" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })
$env:OMP_NUM_THREADS = "$Threads"

$trainArgs = @(
    "-m", "unisharp.cli", "train-feature",
    "--device", "cpu", "--renderer-backend", "portable",
    "--portable-renderer-max-gaussians", "16384", "--portable-renderer-max-gaussians-per-tile", "96",
    "--out-root", $OutRoot, "--run-name", $RunName,
    "--steps", "$Steps", "--batch-size", "1", "--num-workers", "$NumWorkers",
    "--pinhole-train-height", "$Height", "--pinhole-train-width", "$Width", "--train-resize-multiple", "0",
    "--unik3d-backbone", "vitl", "--initializer-stride", "2",
    "--init-checkpoint", $InitCheckpoint, "--no-init-checkpoint-strict",
    "--target-rig-conditioning", "--target-rig-embedding-dim", "$TargetRigEmbedDim",
    "--target-rig-translation-scale", "$TargetRigTranslationScale",
    "--save-every", "$([Math]::Max(1, $Steps))", "--log-every", "1", "--vis-every", "0", "--lambda-percep", "0",
    "--dataset-weight-re10k", "1", "--dataset-weight-hm3d", "0", "--dataset-weight-sim", "0",
    "--dataset-weight-wildrgbd", "0", "--dataset-weight-dl3dv", "0", "--dataset-weight-scanetpp", "0",
    "--data-root-re10k", $DataRootRe10K, "--dataset-manifest-dir", $DatasetManifestDir
)

& python @trainArgs @ExtraArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
