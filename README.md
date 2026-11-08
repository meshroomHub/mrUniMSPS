# mrUniMSPS

Meshroom plugin for [Uni-MS-PS](https://github.com/meshroomHubWarehouse/Uni-MS-PS): universal photometric stereo with a
multi-scale encoder-decoder transformer (C. Hardy, Y. Quéau, D. Tschumperlé).

Uni-MS-PS estimates per-pixel surface normals from multiple images of a scene captured from the same viewpoint under
varying illumination. The node is harmonized with the other photometric stereo plugins (mrLINOUniPS, mrSDMUniPS,
mrLedGetter, through the shared `psCommon.py` layer), and its outputs are ready for RNb-NeuS2:

| Node | Step | Output |
|------|------|--------|
| `UniMSPS` | Normals per pose | normal maps, masks, `normalMaps.sfm` |

---

## Requirements

- **[Meshroom](https://github.com/alicevision/Meshroom)** 2026.1 or later (develop branch until its release, tested on
  commit [`edc4b6a2`](https://github.com/alicevision/Meshroom/commit/edc4b6a2)): Meshroom 2025.1.x has neither the plugin
  manager nor the `[tool.meshroom]` settings of `pyproject.toml`
- **NVIDIA GPU** with a driver supporting CUDA 12.1
- **Python** 3.10 or newer

All Python dependencies are declared in [`pyproject.toml`](pyproject.toml): PyTorch (CUDA 12.1) and `uni-ms-ps`, pinned to the
`weights-v1` tag of [meshroomHubWarehouse/Uni-MS-PS](https://github.com/meshroomHubWarehouse/Uni-MS-PS), a fork of
[Clement-Hardy/Uni-MS-PS](https://github.com/Clement-Hardy/Uni-MS-PS).

---

## Installation

### 1. Install the plugin

From Meshroom's plugin manager (*Manage Plugins*), or from the command line:

```bash
meshroom_plugins install mrUniMSPS
```

The installer creates the plugin `venv/` with `uv sync` and downloads the uncalibrated model weights (~317 MB), declared as assets in `pyproject.toml`, checking their SHA-256.
The `UniMSPS` node then appears in the *Photometric Stereo* category.

### 2. Manual install (without the plugin manager)

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd /path/to/plugins
git clone https://github.com/meshroomHub/mrUniMSPS.git
cd mrUniMSPS
UV_PROJECT_ENVIRONMENT=venv uv sync   # Meshroom looks for venv/ at the plugin root

mkdir -p weights
curl -L -o weights/model_uncalibrated.pth \
    https://github.com/meshroomHubWarehouse/Uni-MS-PS/releases/download/weights-v1/model_uncalibrated.pth
echo "314d2cd36a5e84ec5eddac32cc17df987981574b43b976d6aa7048124162fc21  weights/model_uncalibrated.pth" | sha256sum -c

export MESHROOM_PLUGINS_PATH=/path/to/plugins/mrUniMSPS:$MESHROOM_PLUGINS_PATH
```

On Windows (PowerShell): `$env:UV_PROJECT_ENVIRONMENT = "venv"; uv sync`, then
`$env:MESHROOM_PLUGINS_PATH = "C:\path\to\plugins\mrUniMSPS"`.

The weights mirror those linked from the [Uni-MS-PS](https://github.com/Clement-Hardy/Uni-MS-PS) README.

---

## Pipeline template

`meshroom/uniMsPs.mgt` (*File > New Pipeline > Uni Ms Ps*) runs the structure-from-motion of AliceVision, segments the
object with SAM 3, exports the undistorted images and estimates the normals with `UniMSPS`.

It needs [mrSegmentation](https://github.com/meshroomHub/mrSegmentation) (node `ImageSegmentationSam3`), not in the plugin
registry yet: install it by hand.

CameraInit groups the images of a folder into one pose only if the folder name starts with `PS_`. Elsewhere, each image
becomes its own pose and the photometric stereo nodes find no multi-lighting pose, so images must follow this layout:

```
dataset/
└── PS/
    ├── PS_01/
    │   ├── ambient_all_leds.jpg   # reference image, lit by all the lights
    │   ├── led_01.jpg             # single-light images
    │   ├── led_02.jpg
    │   └── ...
    ├── PS_02/
    └── ...
```

The filename of the reference image must contain `ambient` (CameraInit makes it the pose reference) and `all_leds`
(`SfMFilter_1` keeps these images for the structure from motion and the segmentation).

Drag and drop the parent folder `dataset/` into the Image Gallery of Meshroom, then set the SAM 3 prompt
(`ImageSegmentationSam3_1`, default `stone`) to your object.

---

## Nodes

### `UniMSPS`

The data handling (SfMData, image selection, masks, outputs) is shared with the other photometric stereo plugins
(`psCommon.py`, identical copy in mrLINOUniPS, mrUniMSPS and mrSDMUniPS): the same parameters have the same names,
defaults and behaviour in the three nodes.

#### Inputs

| Parameter | Label | Description |
|-----------|-------|-------------|
| `inputSfm` | SfMData | SfMData with the multi-lighting views (e.g. undistorted images from ExportImages); views sharing a poseId are the lighting images of one pose **(required)** |
| `maskFolder` | Mask Folder | Optional folder of masks named `<poseId>.png` (one per pose) or `<viewId>.png` (one per image, combined by vote); without mask files, masks come from the alpha channels |
| `downscale` | Downscale Factor | Integer downscale factor of the input images, output maps and intrinsics (1-8, default: 1) |
| `nbImages` | Number Of Images | Maximum number of lighting images per pose (-1 = all, default) |

#### Uni-MS-PS options (advanced)

| Parameter | Label | Description |
|-----------|-------|-------------|
| `cropMargin` | Crop Margin | Margin (pixels) around the bounding box of the mask (default: 0, tight box as the original inference); the crop is padded with zeros to a square of side 32 * 2^k |
| `modelPath` | Model | Uncalibrated weights (.pth); if empty: `<plugin>/weights/model_uncalibrated.pth`, then `<Uni-MS-PS>/weights/model_uncalibrated.pth` |
| `uniMsPsPath` | Uni-MS-PS Path | Uni-MS-PS code directory, used if the package is not installed in the plugin environment (default: `${UNI_MS_PS_PATH}`) |

#### Common options (advanced)

| Parameter | Label | Description |
|-----------|-------|-------------|
| `minViewsPerPose` | Min Views Per Pose | Minimum number of views sharing a poseId for a multi-lighting pose (default: 3) |
| `requireLocalizedPoses` | Require Localized Poses | Process only the poses with a camera pose in the SfMData; false for photometric stereo without SfM, e.g. on the CameraInit output (default: true) |
| `imageSelection` | Image Selection | `random` (reproducible, default), `uniform` or `first` images when `nbImages` is lower than the number of images |
| `seed` | Seed | Seed of the image selection and of the network, combined with the poseId (default: 42) |
| `linearizeInput` | Linearize Input | Convert 8/16-bit images from sRGB to linear (default: false) |
| `maskThreshold` | Mask Threshold | Binarization threshold of mask files and alpha channels, as a fraction of the value range (default: 0.5) |
| `maskVoteThreshold` | Mask Vote Threshold | Fraction of the per-image masks a pixel must belong to (default: 0.5, strict majority; 1.0: intersection; 0.0: union) |
| `maskRemoveBorderComponents` | Remove Border Components | Remove the alpha-mask components touching the image border, e.g. the valid area of undistorted images (default: true) |
| `maskUseGlobalFile` | Use Global Mask File | Use `<maskFolder>/mask.png` for the poses without a specific mask file (default: false) |
| `normalConvention` | Normal Convention | `opengl` (x right, y up, z towards the camera, expected by RNb-NeuS2, default) or `opencv` |
| `keepLandmarks` | Keep Landmarks | Keep the 3D landmark positions (without their observations) in the output SfMData, used by RNb-NeuS2 to normalize the scene without masks (default: true) |
| `failurePolicy` | Failure Policy | Fail the node if `noPose` could be processed (default), if `anyPose` failed, or `never` |

#### Settings

| Parameter | Label | Description |
|-----------|-------|-------------|
| `outputFormat` | Output Format | `png16` (16-bit PNG, (n + 1) / 2 * 65535, default) or `exr` (float32 EXR) |
| `useGpu` | Use GPU | Use the GPU for the inference (default: true) |
| `verboseLevel` | Verbose Level | Log verbosity (default: info) |

#### Outputs

| Parameter | Description |
|-----------|-------------|
| `outputFolder` | Folder containing the normal maps (`<poseId>.png` or `.exr`) and the pose masks |
| `outputSfmDataNormal` | SfMData referencing the normal maps (one view per pose, intrinsics scaled by the downscale factor) |
| `outputMaskFolder` | Pose masks (`masks/<poseId>.png`, 0/255): the pixels where a normal is defined |

> **Note:** When no mask folder is provided, the node automatically extracts object masks from input image alpha channels (removing undistortion artifacts via connected component analysis).

---

## Development

### Tests

```bash
# unit tests (CPU), from the plugin root
PYTHONPATH=/path/to/Meshroom venv/bin/python -m pytest tests
# one real pose (GPU): checks sizes, mask coverage and the OpenGL frame of the normals
PYTHONPATH=/path/to/Meshroom venv/bin/python tests/check_real_pose.py <sfm> <poseId> <outputFolder> --downscale 2
```

### Working on a local Uni-MS-PS clone

The node imports Uni-MS-PS from the plugin `venv/`. To run a local clone instead, install it in editable mode in that venv:

```bash
uv pip install --python venv/bin/python -e /path/to/Uni-MS-PS
```

Updating or reinstalling the plugin from the plugin manager recreates `venv/`: run this command again afterwards.

`UNI_MS_PS_PATH` (an environment variable, or the `uniMsPsPath` input of the node) is only a fallback: the node uses it
when Uni-MS-PS is not installed in the venv.

### Plugin structure

```
mrUniMSPS/
├── meshroom/
│   ├── uniMsPs.mgt                # Pipeline template
│   └── UniMSPS/
│       ├── __init__.py
│       ├── UniMSPS.py             # Meshroom node definition
│       └── psCommon.py            # Common photometric stereo data handling (shared copy)
├── tests/                         # Unit tests (pytest) and real pose check (check_real_pose.py)
├── weights/                       # Model weights (Meshroom asset)
│   └── model_uncalibrated.pth
├── venv/                          # Created by the installer or by uv sync
├── pyproject.toml                 # Dependencies and Meshroom settings (uv)
└── README.md
```

For more details on how Meshroom plugins work, see:
- [Meshroom Plugin Install Guide](https://github.com/alicevision/Meshroom/blob/develop/INSTALL_PLUGINS.md)
- [mrHelloWorld](https://github.com/meshroomHub/mrHelloWorld): step-by-step tutorials for building Meshroom plugins

---

## Acknowledgements

This work is supported by [**DOPAMIn**](https://www.cnrsinnovation.com/actualite/une-seconde-promotion-pour-le-programme-open-7-nouveaux-logiciels-scientifiques-a-valoriser/) (*Diffusion Open de Photogrammetrie par AliceVision/Meshroom pour l'Industrie*), selected in the 2024 cohort of the [**OPEN**](https://www.cnrsinnovation.com/open/) programme run by [CNRS Innovation](https://www.cnrsinnovation.com/). OPEN supports the valorization of open-source scientific software by providing dedicated developer resources, governance expertise, and industry partnership support.

**Lead researcher:** [Jean-Denis Durou](https://cv.hal.science/jean-denis-durou), [IRIT](https://www.irit.fr/) (INP-Toulouse)
**Co-lead:** [Lilian Calvet](https://fr.linkedin.com/in/lilian-calvet-42b1a689), [Balgrist University Hospital](https://www.balgrist.ch/)

---

## Related Projects

| Project | Description |
|---------|-------------|
| [Uni-MS-PS](https://github.com/meshroomHubWarehouse/Uni-MS-PS) | Multi-scale transformer for universal photometric stereo |
| [mrLINOUniPS](https://github.com/meshroomHub/mrLINOUniPS) | Meshroom plugin for LINO-UniPS photometric stereo |
| [mrSDMUniPS](https://github.com/meshroomHub/mrSDMUniPS) | Meshroom plugin for SDM-UniPS photometric stereo |
| [mrLedGetter](https://github.com/meshroomHub/mrLedGetter) | Meshroom plugin for LED-getter near-light photometric stereo |
| [mrRNbNeuS2](https://github.com/meshroomHub/mrRNbNeuS2) | Meshroom plugin for RNb-NeuS2 neural surface reconstruction |
| [mrOpenRNb](https://github.com/meshroomHub/mrOpenRNb) | Meshroom plugin for Open-RNb neural surface reconstruction |

---

## Citation

- **Uni MS-PS: a Multi-Scale Encoder Decoder Transformer for Universal Photometric Stereo**  
  C. Hardy, Y. Quéau, D. Tschumperlé  
  [Code](https://github.com/Clement-Hardy/Uni-MS-PS)

---

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE).
