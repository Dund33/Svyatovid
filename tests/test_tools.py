from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


class CreateUserEmbeddingTest(unittest.TestCase):
    def setUp(self):
        self.original_ai_tools = sys.modules.get("ai_tools")
        self.original_tools = sys.modules.pop("tools", None)

        fake_ai_tools = types.ModuleType("ai_tools")
        fake_ai_tools.face_embeddings_from_image = self._face_embeddings_from_image
        sys.modules["ai_tools"] = fake_ai_tools

        self.tools = importlib.import_module("tools")

    def tearDown(self):
        sys.modules.pop("tools", None)
        if self.original_tools is not None:
            sys.modules["tools"] = self.original_tools

        sys.modules.pop("ai_tools", None)
        if self.original_ai_tools is not None:
            sys.modules["ai_tools"] = self.original_ai_tools

    def test_uses_photo_quality_scores_as_medoid_weights(self):
        embedding = self.tools.create_user_embedding_from_images(
            [Path("low_quality.jpg"), Path("high_quality.jpg")],
            photo_quality_service=FakePhotoQualityService(),
        )

        np.testing.assert_allclose(embedding, np.array([0.8, 0.6], dtype=np.float32))

    @staticmethod
    def _face_embeddings_from_image(image_path):
        embeddings = {
            "low_quality.jpg": [np.array([1.0, 0.0], dtype=np.float32)],
            "high_quality.jpg": [np.array([0.8, 0.6], dtype=np.float32)],
        }

        return embeddings[Path(image_path).name]


class FakePhotoQualityService:
    def assess(self, image_path):
        scores = {
            "low_quality.jpg": 0.1,
            "high_quality.jpg": 0.9,
        }

        return SimpleNamespace(score=scores[Path(image_path).name])
