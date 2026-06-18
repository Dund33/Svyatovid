from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from math_tools import largest_cosine_similarity_component_indices, weighted_medoid


class MathToolsTest(unittest.TestCase):
    def test_weighted_medoid_prefers_high_weight_vector(self):
        vectors = [
            np.array([0.0]),
            np.array([10.0]),
            np.array([11.0]),
        ]

        result = weighted_medoid(
            vectors,
            weights=[10.0, 1.0, 1.0],
            metric="euclidean",
        )

        np.testing.assert_array_equal(result, vectors[0])

    def test_weighted_medoid_rejects_invalid_weights(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            weighted_medoid(
                [np.array([1.0]), np.array([2.0])],
                weights=[1.0],
            )

        with self.assertRaisesRegex(ValueError, "negative"):
            weighted_medoid(
                [np.array([1.0]), np.array([2.0])],
                weights=[1.0, -1.0],
            )

    def test_largest_cosine_similarity_component_indices(self):
        vectors = [
            np.array([1.0, 0.0]),
            np.array([0.95, 0.05]),
            np.array([-1.0, 0.0]),
        ]

        indices = largest_cosine_similarity_component_indices(
            vectors,
            similarity_threshold=0.9,
        )

        self.assertEqual(indices, [0, 1])
