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
pip install -r requirements.txt

deactivate
```

This installs Uni-MS-PS and all its dependencies automatically via pip.

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

### Inputs

| Parameter | Label | Description |
|-----------|-------|-------------|
| `inputSfm` | Input SfMData | SfMData JSON file with multi-lighting views grouped by poseId **(required)** |
| `maskFolder` | Mask Folder | Folder with mask PNGs named by poseId or viewId |
| `downscale` | Downscale Factor | Integer downscale factor for input images (1-8, default: 1) |
| `nbImages` | Number of Images | Number of lighting images per pose (-1 = all) |
| `useGpu` | Use GPU | Use GPU for inference (default: true) |

### Outputs

| Parameter | Description |
|-----------|-------------|
| `outputFolder` | Folder containing normal map PNGs |
| `outputSfmDataNormal` | SfMData file referencing normal maps |

---

## Advanced: Developer Setup

If you prefer to work from a local Uni-MS-PS clone instead of pip install:

1. Clone the repo: `git clone -b meshroom https://github.com/meshroomHubWarehouse/Uni-MS-PS.git`
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
│       └── UniMSPS.py             # Meshroom node definition
├── weights/                       # Downloaded model weights
│   └── model_uncalibrated.pth
├── venv/                          # Python virtual environment
├── download_weights.sh            # Weight download script
├── requirements.txt               # Python dependencies (pip install from git)
└── README.md
```

For more details on how Meshroom plugins work, see:
- [Meshroom Plugin Install Guide](https://github.com/alicevision/Meshroom/blob/develop/INSTALL_PLUGINS.md)
- [mrHelloWorld](https://github.com/meshroomHub/mrHelloWorld): step-by-step tutorials for building Meshroom plugins

---

## Acknowledgements

This work is supported by [**DOPAMIn**](https://www.cnrsinnovation.com/actualite/une-seconde-promotion-pour-le-programme-open-7-nouveaux-logiciels-scientifiques-a-valoriser/) (*Diffusion Open de Photogrammetrie par AliceVision/Meshroom pour l'Industrie*), selected in the 2024 cohort of the [**OPEN**](https://www.cnrsinnovation.com/open/) programme run by [CNRS Innovation](https://www.cnrsinnovation.com/). OPEN supports the valorization of open-source scientific software by providing dedicated developer resources, governance expertise, and industry partnership support.

**Lead researcher:** [Jean-Denis Durou](https://cv.hal.science/jean-denis-durou), [IRIT](https://www.irit.fr/) (INP-Toulouse)

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
