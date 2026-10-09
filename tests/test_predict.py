"""Tests of the Uni-MS-PS predict API (meshroom_predict, installed in the plugin environment), CPU only.

Skipped when the Uni-MS-PS package or the weights are not available. Run from the plugin root:
    PYTHONPATH=<Meshroom> venv/bin/python -m pytest tests
"""
import os

# CPU only: keep the GPU free for the inference jobs (must be set before torch initializes CUDA)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

api = pytest.importorskip("meshroom_predict")
torch = pytest.importorskip("torch")

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights", "model_uncalibrated.pth")


@pytest.fixture(scope="module")
def predictor():
    if not os.path.isfile(WEIGHTS):
        pytest.skip("Uni-MS-PS weights not downloaded")
    return api.loadModel(WEIGHTS, useGpu=False)


def sphere(height, width):
    """Normals (OpenGL camera frame) and mask of a sphere, off-center in the image."""
    radius = 0.4 * min(height, width)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (xs - (width / 2.0 + 3)) / radius
    y = -(ys - (height / 2.0 - 2)) / radius
    mask = x * x + y * y < 0.97
    normal = np.stack([x, y, np.sqrt(np.clip(1 - x * x - y * y, 0, 1))], axis=2) * mask[:, :, None]
    return normal.astype(np.float32), mask


def render(normal, mask, nbLights=12):
    """Lambertian images (RGB albedo) under random directional lights."""
    rng = np.random.default_rng(0)
    images = []
    for _ in range(nbLights):
        light = rng.normal(size=3)
        light[2] = abs(light[2]) + 1.0
        light /= np.linalg.norm(light)
        shading = np.clip(normal @ light, 0, None)[:, :, None]
        images.append((shading * np.array([0.8, 0.6, 0.4], np.float32) * mask[:, :, None]).astype(np.float32))
    return images


def meanAngle(a, b, mask):
    cosine = np.clip(np.sum(a[mask] * b[mask], axis=1), -1, 1)
    return float(np.degrees(np.arccos(cosine)).mean())


def test_canvas_size():
    assert api.canvasSize(10, 20) == (32, 1)
    assert api.canvasSize(32, 32) == (32, 1)
    assert api.canvasSize(33, 5) == (64, 2)
    assert api.canvasSize(700, 1024) == (1024, 6)


def test_bounding_box_inclusive_with_margin():
    mask = np.zeros((10, 12), bool)
    mask[2, 3] = mask[6, 9] = True
    assert api.maskBoundingBox(mask, 0) == (2, 7, 3, 10)
    assert api.maskBoundingBox(mask, 3) == (0, 10, 0, 12)
    with pytest.raises(ValueError):
        api.maskBoundingBox(np.zeros((4, 4), bool), 0)


def test_load_model_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        api.loadModel(str(tmp_path / "missing.pth"), useGpu=False)
    invalid = str(tmp_path / "invalid.pth")
    torch.save({"unrelated.weight": torch.zeros(1)}, invalid)
    with pytest.raises(RuntimeError, match="missing weights"):
        api.loadModel(invalid, useGpu=False)


@pytest.mark.parametrize("shape", [(96, 128), (128, 96)])
def test_sphere_opengl_frame(predictor, shape):
    """Landscape and portrait images: OpenGL normals at the input resolution, zero outside the mask."""
    normal, mask = sphere(*shape)
    predicted = api.predict(predictor, render(normal, mask), mask)["normal"]
    assert predicted.shape == shape + (3,) and predicted.dtype == np.float32
    assert not predicted[~mask].any()
    assert np.allclose(np.linalg.norm(predicted[mask], axis=1), 1, atol=1e-4)
    assert meanAngle(predicted, normal, mask) < 10.0


def test_input_value_scale_invariance(predictor):
    """8-bit-like and float values give the same normals (per-image normalization, no 8-bit conversion)."""
    normal, mask = sphere(64, 80)
    images = render(normal, mask, nbLights=8)
    a = api.predict(predictor, images, mask)["normal"]
    b = api.predict(predictor, [image * 65535.0 for image in images], mask)["normal"]
    assert meanAngle(a, b, mask) < 0.1


def test_invalid_inputs(predictor):
    normal, mask = sphere(40, 40)
    images = render(normal, mask, nbLights=3)
    with pytest.raises(ValueError):
        api.predict(predictor, images, np.zeros_like(mask))
    with pytest.raises(ValueError):
        api.predict(predictor, [images[0][:-1]], mask)
    nan = images[0].copy()
    nan[mask] = np.nan
    with pytest.raises(ValueError):
        api.predict(predictor, [nan], mask)
