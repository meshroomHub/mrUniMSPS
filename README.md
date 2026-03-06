<div align="center">

# mrUniMSPS

### Meshroom Plugin for Uni-MS-PS

<p>
Integrate <a href="https://github.com/Clement-Hardy/Uni-MS-PS">Uni-MS-PS</a> multi-view photometric stereo normal estimation directly into your <a href="https://github.com/alicevision/Meshroom">Meshroom</a> photogrammetry pipeline.
</p>

<a href="https://github.com/Clement-Hardy/Uni-MS-PS"><img src="https://img.shields.io/badge/Core-Uni--MS--PS-green" alt="Uni-MS-PS" height="25"></a>

</div>

---

## What is Uni-MS-PS?

**Uni-MS-PS** is a unified multi-view and single-view photometric stereo method for surface normal estimation. Given multiple images of a scene captured from the same viewpoint under varying illumination, it estimates per-pixel surface normals. It supports both calibrated (known light directions) and uncalibrated modes.

---

## Requirements

- **Python** 3.10+
- **CUDA** 12.x + NVIDIA GPU
- **[Meshroom](https://github.com/alicevision/Meshroom)** 2025+ (develop branch)

Full dependency list: [`requirements.txt`](requirements.txt)

---

## Quick Start

> **Prerequisite:** a working [Meshroom](https://github.com/alicevision/Meshroom) installation.

### 1. Clone the plugin

```bash
cd /path/to/your/plugins
git clone https://github.com/meshroomHub/mrUniMSPS.git
```

### 2. Clone the Uni-MS-PS core code

```bash
git clone https://github.com/Clement-Hardy/Uni-MS-PS.git
```

> **Note:** for SfMData JSON support, use the `feat/sfm-inference` branch.

### 3. Set up the virtual environment

Meshroom looks for a folder named **`venv`** at the plugin root and uses its Python interpreter to run the node. You have two options:

#### Option A: Symlink an existing venv

If you already have a working virtual environment from the Uni-MS-PS repository, you can simply symlink it:

```bash
cd mrUniMSPS
ln -s /absolute/path/to/Uni-MS-PS/.venv venv
```

#### Option B: Create a fresh venv

```bash
cd mrUniMSPS

# Create the venv (must be named "venv", not ".venv")
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt

deactivate
```

### 4. Configure the plugin

Edit `meshroom/config.json` to point to your Uni-MS-PS clone:

```json
[
    {
        "key": "UNI_MS_PS_PATH",
        "type": "path",
        "value": "/absolute/path/to/Uni-MS-PS"
    }
]
```

### 5. Download weights

Download the pre-trained model weights and place them in the `weights/` directory inside the Uni-MS-PS code directory.

### 6. Register the plugin in Meshroom

Set the `MESHROOM_PLUGINS_PATH` environment variable:

```bash
# Linux
export MESHROOM_PLUGINS_PATH=/path/to/your/plugins/mrUniMSPS:$MESHROOM_PLUGINS_PATH

# Windows
set MESHROOM_PLUGINS_PATH=C:\path\to\mrUniMSPS;%MESHROOM_PLUGINS_PATH%
```

Launch Meshroom: the **UniMSPS** node appears under the **Photometric Stereo** category.

---

## Plugin Structure

```
mrUniMSPS/
├── meshroom/
│   ├── config.json                # Plugin configuration (UNI_MS_PS_PATH)
│   └── UniMSPS/
│       ├── __init__.py
│       └── UniMSPS.py             # Meshroom node definition
├── venv/                          # Python virtual environment (or symlink, see step 3)
├── requirements.txt               # Python dependencies
└── README.md
```

For more details on how Meshroom plugins work, see:
- [Meshroom Plugin Install Guide](https://github.com/alicevision/Meshroom/blob/develop/INSTALL_PLUGINS.md)
- [mrHelloWorld](https://github.com/meshroomHub/mrHelloWorld): step-by-step tutorials for building Meshroom plugins

---

## Node Parameters

### Inputs

| Parameter | Label | Description |
|-----------|-------|-------------|
| `inputSfm` | Input SfMData | SfMData JSON file with multi-lighting views grouped by poseId **(required)** |
| `maskFolder` | Mask Folder | Folder with mask PNGs named by poseId or viewId |
| `downscale` | Downscale Factor | Integer downscale factor for input images (1-8, default: 1) |
| `nbImages` | Number of Images | Number of lighting images per pose (-1 = all) |
| `calibrated` | Calibrated Mode | Use calibrated model with known light directions |
| `useGpu` | Use GPU | Use GPU for inference (default: true) |
| `uniMsPsPath` | Uni-MS-PS Path | Path to Uni-MS-PS code (set via `config.json`) |

### Outputs

| Parameter | Description |
|-----------|-------------|
| `outputFolder` | Folder containing normal map PNGs |
| `outputJson` | JSON file mapping poseIds to normal map paths |

---

## Acknowledgements

This work is supported by [**DOPAMIn**](https://www.cnrsinnovation.com/actualite/une-seconde-promotion-pour-le-programme-open-7-nouveaux-logiciels-scientifiques-a-valoriser/) (*Diffusion Open de Photogrammetrie par AliceVision/Meshroom pour l'Industrie*), selected in the 2024 cohort of the [**OPEN**](https://www.cnrsinnovation.com/open/) programme run by [CNRS Innovation](https://www.cnrsinnovation.com/). OPEN supports the valorization of open-source scientific software by providing dedicated developer resources, governance expertise, and industry partnership support.

**Lead researcher:** [Jean-Denis Durou](https://cv.hal.science/jean-denis-durou), [IRIT](https://www.irit.fr/) (INP-Toulouse)

---

## Related Projects

| Project | Description |
|---------|-------------|
| [Uni-MS-PS](https://github.com/Clement-Hardy/Uni-MS-PS) | Unified multi-view and single-view photometric stereo |
| [mrSDMUniPS](https://github.com/meshroomHub/mrSDMUniPS) | Meshroom plugin for SDM-UniPS photometric stereo |
| [mrOpenRNb](https://github.com/meshroomHub/mrOpenRNb) | Meshroom plugin for neural surface reconstruction from normals |

---

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE).
