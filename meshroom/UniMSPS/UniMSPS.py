__version__ = "2.0"

import os

from meshroom.core import desc

from . import psCommon

WEIGHTS_NAME = "model_uncalibrated.pth"


class UniMSPS(desc.Node):
    """Multi-view photometric stereo normal estimation with Uni-MS-PS."""

    category = "Photometric Stereo"
    gpu = desc.Level.INTENSIVE
    size = desc.DynamicNodeSize("inputSfm")

    documentation = """
Estimate one normal map per multi-lighting pose with Uni-MS-PS (universal photometric stereo: unknown lighting).

**Inputs:** an SfMData where the lighting images of a pose share the same poseId (as created by CameraInit for
multi-lighting folders), e.g. the undistorted images of ExportImages. Poses with fewer than 'Min Views Per Pose'
views (photogrammetry images) are ignored, so a mixed multi-view / multi-light SfMData can be used as is.

**Masks:** from a mask folder (<poseId>.png or <viewId>.png) or from the alpha channel of the images; the
per-image masks of a pose are combined by vote ('Mask Vote Threshold').

**Processing:** the images are cropped around the mask and padded with zeros to a square of side 32 * 2^k (k + 1
resolution stages of the network): the GPU memory grows with the size of the object in the (downscaled) images.

**Outputs:** one normal map per pose (<poseId>.png|exr, OpenGL camera frame by default), the pose masks
(masks/<poseId>.png: pixels with a normal) and an SfMData referencing the normal maps (one view per pose, with the
intrinsics scaled by the downscale factor), ready for RNb-NeuS2.

The data handling (SfMData, image selection, masks, outputs) is common to the LINOUniPS, UniMSPS and SDMUniPS
nodes; see the advanced options.
"""

    inputs = psCommon.inputAttributes() + [
        desc.IntParam(
            name="cropMargin",
            label="Crop Margin",
            description="Margin (pixels) around the bounding box of the mask (0: tight box, as the original "
                        "Uni-MS-PS inference). The crop is padded with zeros to a square of side 32 * 2^k.",
            value=0,
            range=(0, 256, 1),
            advanced=True,
        ),
        desc.File(
            name="modelPath",
            label="Model",
            description="Uni-MS-PS uncalibrated weights (.pth). If empty: <plugin>/weights/" + WEIGHTS_NAME + ", then "
                        "<Uni-MS-PS>/weights/" + WEIGHTS_NAME + ".",
            value="",
            advanced=True,
        ),
        desc.File(
            name="uniMsPsPath",
            label="Uni-MS-PS Path",
            description="Uni-MS-PS code directory, used if the package is not installed in the plugin environment.",
            value="${UNI_MS_PS_PATH}",
            advanced=True,
            invalidate=False,
        ),
    ] + psCommon.advancedInputAttributes() + psCommon.settingsAttributes()

    outputs = psCommon.outputAttributes()

    @staticmethod
    def findWeights(node):
        if node.modelPath.value:
            return node.modelPath.value
        candidates = [os.path.join(os.path.dirname(__file__), "..", "..", "weights", WEIGHTS_NAME)]
        if node.uniMsPsPath.evalValue:
            candidates.append(os.path.join(node.uniMsPsPath.evalValue, "weights", WEIGHTS_NAME))
        for path in candidates:
            if os.path.isfile(path):
                return os.path.abspath(path)
        raise RuntimeError("Uni-MS-PS weights not found, set 'Model' or download them (download_weights.sh). "
                           "Searched: {}".format(", ".join(candidates)))

    @staticmethod
    def importApi(node):
        try:
            import meshroom_predict
        except ImportError:
            import sys
            path = node.uniMsPsPath.evalValue
            if not path or not os.path.isdir(path):
                raise RuntimeError("Uni-MS-PS is not installed in the plugin environment and 'Uni-MS-PS Path' is "
                                   "invalid: '{}'".format(path))
            sys.path.insert(0, path)
            import meshroom_predict
        return meshroom_predict

    def processChunk(self, chunk):
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            import torch
            node = chunk.node
            api = self.importApi(node)
            weights = self.findWeights(node)
            chunk.logger.info("Uni-MS-PS weights: {}".format(weights))
            predictor = api.loadModel(weights, useGpu=node.useGpu.value, logger=chunk.logger)

            def predict(images, mask):
                return api.predict(predictor, images, mask, cropMargin=node.cropMargin.value)

            def cleanup():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            psCommon.processPoses(chunk, predict, cleanup=cleanup)
        finally:
            chunk.logManager.end()
