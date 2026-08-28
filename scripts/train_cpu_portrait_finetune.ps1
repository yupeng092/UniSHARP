<#
Low-resolution CPU smoke-test fine-tuning on Windows PowerShell.
Loads the released UniSHARP ViT-L checkpoint and uses the PyTorch portable
renderer. This verifies the local training path; use the NPU scripts for a
full fine-tuning run.
#>
[CmdletBinding()]
param(
    [string]$DatasetManifestDir = $env:DATASET_MANIFEST_DIR,
    [string]$FineTuneCheckpoint = "",
    [string]$OutRoot = "",
    [string]$RunName = "",
    [int]$Steps = 10,
    [int]$Threads = 8,
    [int]$NumWorkers = 2,
    [int]$Height = 256,
    [int]$Width = 384,
    [int]$PortableMaxGaussians = 16384,
    [double]$CocoPersonWeight = 0.0,
    [string]$CocoPersonManifest = "",
    [double]$OpenImagesPersonWeight = 0.0,
    [string]$OpenImagesPersonManifest = "",
    [double]$LocalImagesWeight = 0.0,
    [string]$LocalImagesManifest = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($DatasetManifestDir)) {
    throw "Set -DatasetManifestDir or DATASET_MANIFEST_DIR."
}
$DatasetManifestDir = (Resolve-Path $DatasetManifestDir).Path

if ([string]::IsNullOrWhiteSpace($FineTuneCheckpoint)) {
    $FineTuneCheckpoint = Join-Path $repoRoot "checkpoints\released\pretained_model.pt"
}
if (-not (Test-Path -LiteralPath $FineTuneCheckpoint -PathType Leaf)) {
    throw "Official UniSHARP checkpoint was not found: $FineTuneCheckpoint"
}
if ([string]::IsNullOrWhiteSpace($OutRoot)) {
    $OutRoot = Join-Path $repoRoot "outputs_cpu"
}
if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "unisharp_cpu_finetune_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}
if ([string]::IsNullOrWhiteSpace($CocoPersonManifest)) {
    $CocoPersonManifest = Join-Path $DatasetManifestDir "coco_person_train2017_boxes.jsonl"
}
if ([string]::IsNullOrWhiteSpace($OpenImagesPersonManifest)) {
    $OpenImagesPersonManifest = Join-Path $DatasetManifestDir "openimages_person_train_boxes.jsonl"
}
if ([string]::IsNullOrWhiteSpace($LocalImagesManifest)) {
    $LocalImagesManifest = Join-Path $DatasetManifestDir "local_images.txt"
}

$datasetArgs = @()
function Add-WeightedDataset([double]$Weight, [string]$Manifest, [string]$ManifestFlag, [string]$WeightFlag, [string]$Label) {
    if ($Weight -gt 0.0) {
        if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
            throw "$Label manifest was not found: $Manifest"
        }
        $script:datasetArgs += @($ManifestFlag, $Manifest, $WeightFlag, "$Weight")
    }
}
Add-WeightedDataset $LocalImagesWeight $LocalImagesManifest "--local-images-manifest" "--dataset-weight-local-images" "Local images"
Add-WeightedDataset $CocoPersonWeight $CocoPersonManifest "--coco-person-manifest" "--dataset-weight-coco-person" "COCO Person"
Add-WeightedDataset $OpenImagesPersonWeight $OpenImagesPersonManifest "--openimages-person-manifest" "--dataset-weight-openimages-person" "OpenImages Person"
if ($datasetArgs.Count -eq 0) {
    throw "Set at least one positive dataset weight: -CocoPersonWeight, -OpenImagesPersonWeight, or -LocalImagesWeight."
}

$env:PYTHONPATH = "$repoRoot;$repoRoot\UniK3D" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })
$env:OMP_NUM_THREADS = "$Threads"

$trainArgs = @(
    "-m", "unisharp.cli", "train-feature",
    "--device", "cpu", "--renderer-backend", "portable",
    "--portable-renderer-max-gaussians", "$PortableMaxGaussians",
    "--portable-renderer-max-gaussians-per-tile", "96",
    "--out-root", $OutRoot, "--run-name", $RunName,
    "--steps", "$Steps", "--batch-size", "1", "--num-workers", "$NumWorkers",
    "--pinhole-train-height", "$Height", "--pinhole-train-width", "$Width", "--train-resize-multiple", "0",
    # Match the released UniSHARP checkpoint's Gaussian grid density.  A
    # stride of 2 would reduce the same-resolution output to one quarter.
    "--unik3d-backbone", "vitl", "--initializer-stride", "1",
    "--init-checkpoint", $FineTuneCheckpoint, "--init-checkpoint-strict",
    "--unik3d-progressive-unfreeze", "--unik3d-decoder-unfreeze-step", "5",
    "--unik3d-encoder-unfreeze-step", "15", "--unik3d-encoder-last-n-blocks", "4",
    "--lr0", "2e-5", "--lr1", "2e-6", "--unik3d-lr0", "5e-6", "--unik3d-lr1", "5e-7",
    "--unik3d-encoder-lr0", "5e-7", "--unik3d-encoder-lr1", "5e-8",
    "--save-every", "$([Math]::Max(1, $Steps))", "--log-every", "1", "--vis-every", "0", "--lambda-percep", "0",
    "--dataset-weight-re10k", "0", "--dataset-weight-hm3d", "0", "--dataset-weight-sim", "0",
    "--dataset-weight-wildrgbd", "0", "--dataset-weight-dl3dv", "0", "--dataset-weight-scanetpp", "0"
)

& python @trainArgs @datasetArgs @ExtraArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
