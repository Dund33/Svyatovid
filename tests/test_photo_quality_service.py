from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from services.PhotoQualityService import PhotoQualityService


class PhotoQualityServiceTest(unittest.TestCase):
    def setUp(self):
        self.service = PhotoQualityService()
        self.image = _synthetic_photo()

    def test_assesses_image_from_path(self):
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "photo.jpg"
            self.image.save(image_path, quality=95)

            assessment = self.service.assess(image_path)

        self.assertEqual(assessment.width, self.image.width)
        self.assertEqual(assessment.height, self.image.height)
        self.assertGreaterEqual(assessment.score, 0.0)
        self.assertLessEqual(assessment.score, 1.0)

    def test_noise_increases_noise_severity(self):
        clean = self.service.assess(self.image)
        noisy = self.service.assess(_add_noise(self.image, sigma=35))

        self.assertGreater(noisy.noise.severity, clean.noise.severity + 0.2)
        self.assertLess(noisy.score, clean.score)

    def test_nearest_resize_increases_pixelization_severity(self):
        clean = self.service.assess(self.image)
        pixelated = self.service.assess(_pixelated(self.image))

        self.assertGreater(
            pixelated.pixelization.severity,
            clean.pixelization.severity + 0.15,
        )
        self.assertLess(pixelated.score, clean.score)

    def test_bicubic_resize_increases_upscaling_artifact_severity(self):
        clean = self.service.assess(self.image)
        upscaled = self.service.assess(_bicubic_upscaled(self.image))

        self.assertGreater(
            upscaled.upscaling_artifacts.severity,
            clean.upscaling_artifacts.severity + 0.05,
        )
        self.assertLess(upscaled.score, clean.score)


def _synthetic_photo(size: tuple[int, int] = (160, 160)) -> Image.Image:
    width, height = size
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)

    red = 0.48 + 0.24 * np.sin(x / 7.0) + 0.12 * np.cos((x + y) / 13.0)
    green = 0.48 + 0.20 * np.sin(y / 9.0) + 0.14 * np.cos((x - y) / 17.0)
    blue = 0.48 + 0.18 * np.sin((x + 2 * y) / 11.0)
    image = np.stack([red, green, blue], axis=2)

    image[32:112, 44:118, 0] += 0.17
    image[60:92, 20:140, 1] -= 0.12
    image[24:132, 78:86, 2] += 0.20

    image = np.clip(image, 0.0, 1.0)

    return Image.fromarray((image * 255).astype(np.uint8), mode="RGB")


def _add_noise(image: Image.Image, sigma: float) -> Image.Image:
    rng = np.random.default_rng(4)
    array = np.asarray(image, dtype=np.float32)
    noisy = array + rng.normal(0.0, sigma, size=array.shape)

    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="RGB")


def _pixelated(image: Image.Image) -> Image.Image:
    small = image.resize((40, 40), Image.Resampling.BOX)

    return small.resize(image.size, Image.Resampling.NEAREST)


def _bicubic_upscaled(image: Image.Image) -> Image.Image:
    small = image.resize((40, 40), Image.Resampling.BICUBIC)

    return small.resize(image.size, Image.Resampling.BICUBIC)
