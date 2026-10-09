<div align="center">

# mrUniMSPS

### Meshroom Plugin for Uni-MS-PS

<p>
Integrate <a href="https://github.com/meshroomHubWarehouse/Uni-MS-PS">Uni-MS-PS</a> multi-view photometric stereo normal estimation directly into your <a href="https://github.com/alicevision/Meshroom">Meshroom</a> photogrammetry pipeline.
</p>

<a href="https://github.com/meshroomHubWarehouse/Uni-MS-PS"><img src="https://img.shields.io/badge/Core-Uni--MS--PS-green" alt="Uni-MS-PS" height="25"></a>

</div>

---

## What is Uni-MS-PS?

**Uni-MS-PS** is a unified multi-view and single-view photometric stereo method for surface normal estimation. Given multiple images of a scene captured from the same viewpoint under varying illumination, it estimates per-pixel surface normals. It supports both calibrated (known light directions) and uncalibrated modes.

---

## Requirements

- **Python** 3.10+
- **CUDA** 12.x + NVIDIA GPU
- **[Meshroom](https://github.com/alicevision/Meshroom)** 2025+ (develop branch)

---

## Quick Start

> **Prerequisite:** a working [Meshroom](https://github.com/alicevision/Meshroom) installation.

### 1. Clone the plugin

```bash
cd /path/to/your/plugins
git clone https://github.com/meshroomHub/mrUniMSPS.git
cd mrUniMSPS
```

### 2. Set up the virtual environment

Meshroom looks for a folder named **`venv`** at the plugin root.

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch torchvision
pip install -e .

deactivate
```

The plugin's `pyproject.toml` installs Uni-MS-PS and its dependencies. Meshroom
still discovers the node from this repository via `MESHROOM_PLUGINS_PATH`;
`requirements.txt` is retained for older installation workflows.

### 3. Download pretrained weights

```bash
bash download_weights.sh
```

This downloads the uncalibrated model weights (~303 MB) from Google Drive into `weights/`:

```
weights/
└── model_uncalibrated.pth
```

The plugin auto-detects this directory. No config.json needed.

> **Note:** if the download fails (Google Drive quota), install [gdown](https://github.com/wkentaro/gdown) (`pip install gdown`) and re-run the script.

### 4. Register the plugin in Meshroom

```bash
export MESHROOM_PLUGINS_PATH=/path/to/your/plugins/mrUniMSPS:$MESHROOM_PLUGINS_PATH
```

Launch Meshroom: the **UniMSPS** node appears under **Photometric Stereo**.

---

## Node Parameters

The data handling (SfMData, image selection, masks, outputs) is shared with the other photometric stereo plugins
(`psCommon.py`, identical copy in mrLINOUniPS, mrUniMSPS and mrSDMUniPS): the same parameters have the same names,
defaults and behaviour in the three nodes.

### Inputs

| Parameter | Label | Description |
|-----------|-------|-------------|
| `inputSfm` | SfMData | SfMData with the multi-lighting views (e.g. undistorted images from ExportImages); views sharing a poseId are the lighting images of one pose **(required)** |
| `maskFolder` | Mask Folder | Optional folder of masks named `<poseId>.png` (one per pose) or `<viewId>.png` (one per image, combined by vote); without mask files, masks come from the alpha channels |
| `downscale` | Downscale Factor | Integer downscale factor of the input images, output maps and intrinsics (1-8, default: 1) |
| `nbImages` | Number Of Images | Maximum number of lighting images per pose (-1 = all, default) |

### Uni-MS-PS options (advanced)

| Parameter | Label | Description |
|-----------|-------|-------------|
| `cropMargin` | Crop Margin | Margin (pixels) around the bounding box of the mask (default: 0, tight box as the original inference); the crop is padded with zeros to a square of side 32 * 2^k |
| `modelPath` | Model | Uncalibrated weights (.pth); if empty: `<plugin>/weights/model_uncalibrated.pth`, then `<Uni-MS-PS>/weights/model_uncalibrated.pth` |
| `uniMsPsPath` | Uni-MS-PS Path | Uni-MS-PS code directory, used if the package is not installed in the plugin environment (default: `${UNI_MS_PS_PATH}`) |

### Common options (advanced)

| Parameter | Label | Description |
|-----------|-------|-------------|
| `minViewsPerPose` | Min Views Per Pose | Minimum number of views sharing a poseId for a multi-lighting pose (default: 3) |
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

### Settings

| Parameter | Label | Description |
|-----------|-------|-------------|
| `outputFormat` | Output Format | `png16` (16-bit PNG, (n + 1) / 2 * 65535, default) or `exr` (float32 EXR) |
| `useGpu` | Use GPU | Use the GPU for the inference (default: true) |
| `verboseLevel` | Verbose Level | Log verbosity (default: info) |

### Outputs

| Parameter | Description |
|-----------|-------------|
| `outputFolder` | Folder containing the normal maps (`<poseId>.png` or `.exr`) and the pose masks |
| `outputSfmDataNormal` | SfMData referencing the normal maps (one view per pose, intrinsics scaled by the downscale factor) |
| `outputMaskFolder` | Pose masks (`masks/<poseId>.png`, 0/255): the pixels where a normal is defined |

> **Note:** When no mask folder is provided, the node automatically extracts object masks from input image alpha channels (removing undistortion artifacts via connected component analysis).

---

## Advanced: Developer Setup

If you prefer to work from a local Uni-MS-PS clone instead of pip install:

1. Clone the repo: `git clone https://github.com/meshroomHubWarehouse/Uni-MS-PS.git`
2. Edit `meshroom/config.json`:
   ```json
   [
       {"key": "UNI_MS_PS_PATH", "type": "path", "value": "/path/to/Uni-MS-PS"}
   ]
   ```
3. Place weights in the `weights/` directory of the Uni-MS-PS clone.

The node tries pip imports first, then falls back to the config path.

---

## Plugin Structure

```
mrUniMSPS/
├── meshroom/
│   ├── config.json                # Plugin configuration (optional for dev)
│   └── UniMSPS/
│       ├── __init__.py
│       ├── UniMSPS.py             # Meshroom node definition
│       └── psCommon.py            # Common photometric stereo data handling (shared copy)
├── tests/                         # Unit tests (pytest) and real pose check (check_real_pose.py)
├── weights/                       # Downloaded model weights
│   └── model_uncalibrated.pth
├── venv/                          # Python virtual environment
├── download_weights.sh            # Weight download script
├── pyproject.toml                 # Plugin dependency metadata
├── requirements.txt               # Legacy dependency list
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
| [Uni-MS-PS](https://github.com/meshroomHubWarehouse/Uni-MS-PS) | Unified multi-view and single-view photometric stereo |
| [mrSDMUniPS](https://github.com/meshroomHub/mrSDMUniPS) | Meshroom plugin for SDM-UniPS photometric stereo |
| [mrOpenRNb](https://github.com/meshroomHub/mrOpenRNb) | Meshroom plugin for neural surface reconstruction from normals |

---

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE).
