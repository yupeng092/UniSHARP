<h1 align="center">
UniSHARP:<br>
Universal Sharp Monocular View Synthesis
</h1>

<p align="center">
  <b>Meixi Song</b><sup>1</sup> ·
  <b>Dizhe Zhang</b><sup>1,*</sup> ·
  <b>Hao Ren</b><sup>1,2</sup> ·
  <b>Ruiyang Zhang</b><sup>1,3</sup> ·
  <b>Bo Du</b><sup>4</sup> ·
  <b>Ming-Hsuan Yang</b><sup>5</sup> ·
  <b>Lu Qi</b><sup>1,4,*</sup>
  <br>
  <sup>1</sup>Insta360 Research · <sup>2</sup>Sun Yat-sen University · <sup>3</sup>Beihang University · <sup>4</sup>Wuhan University · <sup>5</sup>University of California, Merced
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.07514"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href="https://insta360-research-team.github.io/Unisharp-website/"><img src="https://img.shields.io/badge/Project_Page-Website-green" alt="Project Page"></a>
  <a href="https://huggingface.co/spaces/Insta360-Research/UniSHARP"><img src="https://img.shields.io/badge/HuggingFace-Demo-yellow" alt="Demo"></a>
  <a href="https://huggingface.co/datasets/Insta360-Research/OmniRooms"><img src="https://img.shields.io/badge/HuggingFace-OmniRooms-orange" alt="Dataset"></a>
  <a href="https://github.com/Insta360-Research-Team/UniSHARP"><img src="https://img.shields.io/badge/GitHub-Code-blue?logo=github&logoColor=white" alt="GitHub"></a>
</p>

UniSHARP extends SHARP-style photorealistic monocular view synthesis to universal camera systems. Given a single image from a perspective, wide-FoV, fisheye, or panoramic camera, UniSHARP predicts a 3D Gaussian representation and renders high-quality novel views.

<p align="center">
  <img src="assets/teaser.gif" width="59.4%" alt="UniSHARP teaser">
  <img src="assets/teaser2.gif" width="39.6%" alt="UniSHARP teaser 2">
</p>

<p align="center">
  <img src="assets/unisharp.png" alt="UniSHARP method" width="90%">
</p>

## 🔨 Installation

Clone this repository and enter the project directory:

```bash
git clone https://github.com/Insta360-Research-Team/UniSHARP.git
cd Unisharp
```

Create a fresh conda environment:

```bash
conda create -n unisharp python=3.12 -y
conda activate unisharp
```

Install PyTorch for your CUDA version. The code was smoke-tested with PyTorch 2.8 and torchvision 0.23:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

Install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

## 🧩 External Dependencies

### UniK3D

UniSHARP uses UniK3D for universal camera ray and feature prediction. Clone the official repository into `Unisharp/UniK3D`:

```bash
git clone https://github.com/lpiccinelli-eth/UniK3D.git UniK3D
```

### 3DGEER

Fisheye rendering depends on the GEER CUDA rasterizer from 3DGEER. Clone the repository into `Unisharp/3dgeer`:

```bash
git clone https://github.com/boschresearch/3dgeer.git 3dgeer
```

If you only use perspective or panoramic inference, the GEER rasterizer may not be needed. It is required for fisheye rendering paths.

## 🖼️ Dataset

The released dataset is hosted on Hugging Face:

- Dataset: [Insta360-Research/OmniRooms](https://huggingface.co/datasets/Insta360-Research/OmniRooms)
- Training manifests: [Insta360-Research/OmniRooms/manifests/train](https://huggingface.co/datasets/Insta360-Research/OmniRooms/tree/main/manifests/train)
- Validation manifests: [Insta360-Research/OmniRooms/manifests/validation](https://huggingface.co/datasets/Insta360-Research/OmniRooms/tree/main/manifests/validation)

**OmniRooms** is a panoramic simulation dataset highly suitable for 3D reconstruction, especially for 3DGS tasks. It consists of 16 large indoor scenes, each containing multiple rooms, and 300k RGB images covering both small and large pose movements with corresponding depth information. OmniRooms is collected via AirSim, with **OmniRooms-Wide** derived by projecting these panoramas into 130-degree equidistant fisheye views. For each anchor point on a 0.5 m voxel grid, we render one central camera and 29 cameras randomly sampled within a local axis-aligned 30 cm cube centered on the source camera. To isolate translation-induced synthesis, all cameras share a fixed orientation. Each frame is rendered as a 1024 x 2048 ERP image.

<p align="center">
  <img src="assets/dataset/AIUE5_vol8_03_2x2.jpg" width="24%" alt="OmniRooms scene AIUE5 vol8 03">
  <img src="assets/dataset/AIUE5_vol8_04_2x2.jpg" width="24%" alt="OmniRooms scene AIUE5 vol8 04">
  <img src="assets/dataset/AIUE5_vol8_05_2x2.jpg" width="24%" alt="OmniRooms scene AIUE5 vol8 05">
  <img src="assets/dataset/AIUE_V01_001_2x2.jpg" width="24%" alt="OmniRooms scene AIUE V01 001">
  <br>
  <img src="assets/dataset/AIUE_V01_003_2x2.jpg" width="24%" alt="OmniRooms scene AIUE V01 003">
  <img src="assets/dataset/AIUE_V01_004_2x2.jpg" width="24%" alt="OmniRooms scene AIUE V01 004">
  <img src="assets/dataset/AIUE_V02_001_2x2.jpg" width="24%" alt="OmniRooms scene AIUE V02 001">
  <img src="assets/dataset/AI_vol3_01_2x2.jpg" width="24%" alt="OmniRooms scene AI vol3 01">
  <br>
  <img src="assets/dataset/AI_vol3_02_2x2.jpg" width="24%" alt="OmniRooms scene AI vol3 02">
  <img src="assets/dataset/AI_vol3_03_2x2.jpg" width="24%" alt="OmniRooms scene AI vol3 03">
  <img src="assets/dataset/AI_vol3_04_2x2.jpg" width="24%" alt="OmniRooms scene AI vol3 04">
  <img src="assets/dataset/AI_vol4_01_2x2.jpg" width="24%" alt="OmniRooms scene AI vol4 01">
  <br>
  <img src="assets/dataset/AI_vol4_02_2x2.jpg" width="24%" alt="OmniRooms scene AI vol4 02">
  <img src="assets/dataset/AI_vol4_03_2x2.jpg" width="24%" alt="OmniRooms scene AI vol4 03">
  <img src="assets/dataset/AI_vol4_04_2x2.jpg" width="24%" alt="OmniRooms scene AI vol4 04">
  <img src="assets/dataset/AI_vol4_05_2x2.jpg" width="24%" alt="OmniRooms scene AI vol4 05">
</p>

The code supports the following data sources and manifest aliases:

- `RealEstate10K`
- `HM3D`
- `OmniRooms`
- `OmniRooms-Wide`
- `WildRGB-D`
- `DL3DV` 
- `ScanNet++ Fisheye`
- `Replica`, and `Tanks and Temples` for validation-only protocols

Training manifests use the names released under `manifests/train`:

```text
dataset_manifests/
├── re10k_train_chunks.txt            
├── hm3d_train_scenes.txt            
├── omnirooms.txt              
├── wildrgbd_train_scenes.txt         
├── dl3dv_train_scenes.txt            
└── scanetpp_fisheye_train_scenes.txt 
```

Validation manifests use the names released under `manifests/validation`:

```text
validation_manifests/
├── re10k.txt                      
├── dl3dv.txt                         
├── hm3d.txt                          
├── omnirooms.txt                      
├── omnirooms_wide.txt              
├── wildrgbd.txt                     
├── scanetpp_fisheye.txt              
├── replica.txt                       
├── tat.txt                           
```

## 🤝 Checkpoints

Training starts UniSHARP heads from scratch and loads the original pretrained UniK3D weights through the UniK3D loader. The official launcher does not resume from a previous UniSHARP checkpoint by default.

Released UniSHARP checkpoints are available at [Insta360-Research/Unisharp](https://huggingface.co/Insta360-Research/Unisharp/tree/main). Place a checkpoint anywhere on disk and pass the path to validation or inference:

```bash
CHECKPOINT=/path/to/pretained_model.pt
```

## 🚀 Training

Use the official gt-override training launcher:

```bash
bash scripts/train.sh
```

### CPU / Ascend NPU portable pre-training

The portable path uses a differentiable, standard-PyTorch pinhole Gaussian
renderer. It is intended for RE10K/WildRGB-D/DL3DV pre-training and does not
support the CUDA-only panoramic, fisheye, or GEER rendering branches. Set the
dataset and manifest roots first:

```bash
export DATA_ROOT_RE10K=/path/to/re10k
export DATA_ROOT_WILDRGBD=/path/to/wildrgbd
export DATA_ROOT_DL3DV=/path/to/dl3dv_rgb
export DATA_ROOT_DL3DV_DEPTH=/path/to/dl3dv_depth
export DATASET_MANIFEST_DIR=/path/to/dataset_manifests
```

Run a CPU smoke test or low-throughput training:

```bash
bash scripts/train_cpu_portable.sh
```

On an Ascend host with CANN and a matching `torch_npu` installation:

```bash
pip install -r requirements_npu.txt
bash scripts/download_npu_assets.sh
# After accepting the source dataset terms, download selected training data.
HF_TOKEN=hf_your_token bash scripts/download_npu_assets.sh \
  --assets re10k wildrgbd dl3dv \
  --wildrgbd-category <category-or-all> \
  --dl3dv-subset 1K --dl3dv-resolution 960P
# Convert RE10K frames/cameras, then generate DL3DV pseudo z-depth on the NPU.
python scripts/prepare_re10k_chunks.py \
  --source-root /path/to/re10k_frames_and_metadata \
  --output-root "$DATA_ROOT_RE10K" \
  --manifest-path "$DATASET_MANIFEST_DIR/re10k_train_chunks.txt"
python scripts/prepare_dl3dv_unik3d_depth.py \
  --rgb-root "$DATA_ROOT_DL3DV" \
  --depth-root "$DATA_ROOT_DL3DV_DEPTH" \
  --device npu:0 --backbone vits \
  --manifest-path "$DATASET_MANIFEST_DIR/dl3dv_train_scenes.txt"
bash scripts/train_npu_portable.sh
NPU_IDS=0,1,2,3 bash scripts/train_npu_portable_ddp.sh
```

`requirements_npu.txt` documents the non-pip driver/CANN prerequisites and
the required PyTorch/torch_npu version-matching rule. Install that matched
three-package set before running the requirements file.

These launchers use the `vits` UniK3D backbone by default and write a
portable-renderer configuration into each output directory. For offline
multi-view rendering of exported Gaussian scenes, use
`D:/PythonFiles/flash3d-main/render_cpu_multiview.py`; it is an inference
tool, not the differentiable renderer used during training.

The NPU launchers default to a rectangular `1536x1024` pinhole input
(width x height), with `--train-resize-multiple 0`, so every pinhole dataset
is resized to the same geometry before the UniSHARP forward pass. Override
with `PINHOLE_TRAIN_WIDTH` and `PINHOLE_TRAIN_HEIGHT` together. This is much
larger than the old 256-square smoke-test setting: the portable renderer
still follows the full gsplat reference math, so use batch size 1 and expect
substantially lower throughput than the fused CUDA renderer.

#### NPU asset download and resolution policy

The public-asset downloader caches the selected UniK3D weights and copies the
released manifests into the layout expected by this repository:

```bash
# Default: UniK3D-ViT-S weights plus manifests only.
bash scripts/download_npu_assets.sh

# Cache ViT-S and ViT-B, and download a released UniSHARP checkpoint.
bash scripts/download_npu_assets.sh \
  --backbones vits vitb \
  --assets unik3d manifests unisharp-checkpoints

# Explicit only: full public OmniRooms download (about 541 GB).
bash scripts/download_npu_assets.sh --assets omnirooms

# Download official pinhole sources after accepting their respective terms.
# Start with a small WildRGB-D category and the first 1K DL3DV scenes.
HF_TOKEN=hf_your_token bash scripts/download_npu_assets.sh \
  --assets re10k wildrgbd dl3dv \
  --wildrgbd-category <category-name> \
  --dl3dv-subset 1K --dl3dv-resolution 960P
```

Set `HF_TOKEN` before these commands only when the host needs authenticated
Hugging Face access; accepting the RE10K and DL3DV dataset terms is still
required. The downloader invokes WildRGB-D's official downloader, which
requires about 4 TB after extracting every category. DL3DV's official
downloader provides RGB/poses but not the `mini_npz/per_image` depth tree that
this trainer consumes: set `DATA_ROOT_DL3DV_DEPTH` to an already prepared
depth export, then run `prepare_dl3dv_unik3d_depth.py` to create it and its
manifest. The depth script runs pretrained UniK3D with the recorded pinhole
intrinsics and converts radial distance to +Z camera depth before writing one
`depth` array per `.npz`. Likewise, the training loader requires RE10K in its
pre-packed `train/*.torch` format; use `prepare_re10k_chunks.py` after the
official frame-and-camera preparation step. It accepts the public
`images/<scene>/` plus `metadata/<scene>.json` layout and validates every
frame's intrinsics and pose before packing. The
released OmniRooms data is synthetic indoor panorama data; it is not a public
collection of portrait, outdoor-travel, or arbitrary-web videos.

UniK3D-ViT-S and ViT-B share the original dynamic preprocessor: aspect ratio
is constrained to 0.5--2.5, the internal feature image is a multiple of 14,
and its pixel budget is 200k--600k. The current NPU launchers use
`--unik3d-resolution-level 0`, selecting the 200k--240k tier; the outer
`1536x1024` training image is therefore internally resampled to the selected
UniK3D feature tier and aligned back afterwards. This preserves the UniK3D
resolution policy. The portable renderer now follows the
gsplat-classic pinhole mathematics (2D covariance filtering, 3.33-sigma tile
bounds, depth ordering, alpha threshold/cap, and early-transmittance stopping)
without default Gaussian culling. It is a PyTorch reference implementation,
so it is much slower than a fused CUDA/NPU rasterizer.

For an explicit speed or memory trade-off only, set finite limits before the
NPU launcher; this intentionally stops being a complete gsplat reference pass:

```bash
PORTABLE_MAX_GAUSSIANS=8192 PORTABLE_MAX_GAUSSIANS_PER_TILE=128 \
  bash scripts/train_npu_portable.sh
```

The NPU portable renderer currently supports pinhole data only. Its launchers
therefore perform mixed RE10K + WildRGB-D + DL3DV training by default. The
per-dataset weights can be changed with `DATASET_WEIGHT_RE10K`,
`DATASET_WEIGHT_WILDRGBD`, and `DATASET_WEIGHT_DL3DV`. OmniRooms, HM3D, and
ScanNet++ remain disabled until a generic-camera NPU renderer is available.

Only a deterministic part of each already-prepared source is used by default:
10% of RE10K chunks, all scenes in the chosen WildRGB-D category/categories,
and the selected DL3DV subset (normally `1K`). Change these without deleting
or copying source files:

```bash
DATASET_FRACTION_RE10K=0.10 \
DATASET_FRACTION_WILDRGBD=1.0 \
DATASET_FRACTION_DL3DV=1.0 \
  bash scripts/train_npu_portable.sh
```

Training outputs are saved under:

```text
outputs/<run_name>/
├── config.json
├── losses.csv
├── step_XXXXXXX.pt
└── vis/
```

## 📊 Validation

Run validation with a checkpoint:

```bash
bash scripts/validate_unisharp.sh /path/to/step_XXXXXXX.pt
```

## 📒 Inference

Run single-image inference:

```bash
python scripts/infer_unisharp.py \
  --checkpoint /path/to/step_XXXXXXX.pt \
  --image /path/to/image.jpg \
  --out-dir outputs/inference
```

Run a directory or image list:

```bash
python scripts/infer_unisharp.py \
  --checkpoint /path/to/step_XXXXXXX.pt \
  --image-dir /path/to/images \
  --out-dir outputs/inference
```

### CPU inference and multiview rendering

`scripts/infer_unisharp_cpu.py` is the CPU migration of the native inference
path: it keeps the same UniK3D camera-ray estimation, automatic camera fitting,
UniSHARP Gaussian prediction, and export convention, without importing CUDA
rendering dependencies. It writes `gaussians.pt`, `metadata.json`, and optional
`gaussians.ply` for every input image.

For perspective images, add `--render-multiview` to invoke the CPU renderer at
`D:/PythonFiles/flash3d-main/render_cpu_multiview.py` automatically. The result
is saved under each image's `multiview/` directory (views, contact sheet, and
render report):

```bash
python scripts/infer_unisharp_cpu.py \
  --checkpoint /path/to/step_XXXXXXX.pt \
  --image /path/to/perspective.jpg \
  --out-dir outputs/cpu_inference \
  --max-long-edge 384 \
  --threads 8 \
  --render-multiview \
  --render-rig cross5
```

The default Flash3D location can be changed with `--flash3d-root` or the
`FLASH3D_ROOT` environment variable. If Flash3D uses another Python
environment, pass its interpreter through `--renderer-python`.

The configured Flash3D CPU renderer is pinhole-only, so fisheye and panorama
inputs still run the complete CPU prediction/export path but skip multiview
rendering with a warning. Their native CUDA renderer remains
`scripts/infer_unisharp.py`.

If calibrated camera parameters are available, pass them through a JSON file. Without this file, the script predicts rays with UniK3D and fits the camera parameters automatically.

Example perspective camera JSON:

```json
{
  "camera": "perspective",
  "intrinsics": {
    "fx": 820.0,
    "fy": 820.0,
    "cx": 512.0,
    "cy": 384.0
  }
}
```

```bash
python scripts/infer_unisharp.py \
  --checkpoint /path/to/step_XXXXXXX.pt \
  --image /path/to/perspective.jpg \
  --camera-json /path/to/perspective_camera.json
```

Example Fisheye624 camera JSON:

```json
{
  "camera": "fisheye",
  "camera_params": [820.0, 820.0, 512.0, 384.0, 0.01, -0.001, 0.0, 0.0]
}
```

```bash
python scripts/infer_unisharp.py \
  --checkpoint /path/to/step_XXXXXXX.pt \
  --image /path/to/fisheye.jpg \
  --camera-json /path/to/fisheye_camera.json
```

For batched inference, the JSON can also contain per-image entries:

```json
{
  "default": {
    "camera": "perspective",
    "intrinsics": [820.0, 820.0, 512.0, 384.0]
  },
  "images": {
    "panorama.jpg": {
      "camera": "panorama"
    },
    "fisheye.jpg": {
      "camera": "fisheye",
      "camera_params": [820.0, 820.0, 512.0, 384.0, 0.01, -0.001, 0.0, 0.0]
    }
  }
}
```


## 🙏 Acknowledgement

This project builds on open-source work from:

- [SHARP](https://github.com/apple/ml-sharp) for monocular Gaussian view synthesis
- [UniK3D](https://github.com/lpiccinelli-eth/UniK3D) for universal camera geometry and features
- [3DGEER](https://github.com/boschresearch/3dgeer) for generic-camera Gaussian rasterization
- [gsplat](https://github.com/nerfstudio-project/gsplat) for Gaussian splatting utilities

## 📝 Citation

```bibtex
@article{song2026unisharp,
  title={UniSHARP: Universal Sharp Monocular View Synthesis},
  author={Song, Meixi and Zhang, Dizhe and Ren, Hao and Zhang, Ruiyang and Du, Bo and Yang, Ming-Hsuan and Qi, Lu},
  journal={arXiv},
  year={2026}
}
```
