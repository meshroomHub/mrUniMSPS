#!/bin/bash
# Download Uni-MS-PS pretrained weights
#
# Downloads the uncalibrated model weights from Google Drive
# into the weights/ directory of this plugin.
#
# Usage:
#   cd mrUniMSPS
#   bash download_weights.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"
WEIGHTS_FILE="${WEIGHTS_DIR}/model_uncalibrated.pth"

# Google Drive file ID for the uncalibrated model
FILE_ID="1scrTLjQCz0yJPecBMaaCqto2gDFrcCkE"

echo "=== Uni-MS-PS Weights Download ==="

# Check if already downloaded
if [ -f "${WEIGHTS_FILE}" ]; then
    echo "Weights already present: ${WEIGHTS_FILE}"
    echo "  Size: $(du -h "${WEIGHTS_FILE}" | cut -f1)"
    echo "To re-download, remove the file first."
    exit 0
fi

mkdir -p "${WEIGHTS_DIR}"

echo "Downloading weights from Google Drive (~303 MB)..."

# Use gdown if available, otherwise use curl with confirmation bypass
if command -v gdown &> /dev/null; then
    gdown "${FILE_ID}" -O "${WEIGHTS_FILE}"
else
    echo "(gdown not found, using curl — if this fails, install gdown: pip install gdown)"
    # Google Drive large file download requires confirmation token
    CONFIRM=$(curl -sc /tmp/gdrive_cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" | \
              grep -o 'confirm=[^&]*' | head -1 | cut -d= -f2)
    if [ -z "${CONFIRM}" ]; then
        # Small file, direct download
        curl -L -o "${WEIGHTS_FILE}" "https://drive.google.com/uc?export=download&id=${FILE_ID}"
    else
        curl -Lb /tmp/gdrive_cookie -o "${WEIGHTS_FILE}" \
            "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}"
    fi
    rm -f /tmp/gdrive_cookie
fi

# Verify
if [ -f "${WEIGHTS_FILE}" ] && [ -s "${WEIGHTS_FILE}" ]; then
    echo ""
    echo "Download complete:"
    echo "  ${WEIGHTS_FILE} ($(du -h "${WEIGHTS_FILE}" | cut -f1))"
else
    echo "ERROR: Download failed or file is empty."
    echo "Try installing gdown (pip install gdown) and re-running."
    exit 1
fi
