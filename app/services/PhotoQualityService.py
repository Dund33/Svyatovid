from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageOps


ImageInput = str | Path | Image.Image


@dataclass(frozen=True)
class PhotoQualityCriterion:
    name: str
    severity: float
    score: float
    label: str


@dataclass(frozen=True)
class PhotoQualityAssessment:
    score: float
    label: str
    is_acceptable: bool
    width: int
    height: int
    upscaling_artifacts: PhotoQualityCriterion
    pixelization: PhotoQualityCriterion
    noise: PhotoQualityCriterion
    metrics: dict[str, float]
    reasons: list[str]


class PhotoQualityService:
    DEFAULT_WEIGHTS = {
        "upscaling_artifacts": 0.4,
        "pixelization": 0.35,
        "noise": 0.25,
    }

    def __init__(
        self,
        min_acceptable_score: float = 0.65,
        weights: Mapping[str, float] | None = None,
    ):
        self.min_acceptable_score = min_acceptable_score
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)

    def assess(self, image: ImageInput) -> PhotoQualityAssessment:
        rgb = self._load_rgb_image(image)
        gray = self._to_luminance(rgb)
        severities = {
            "upscaling_artifacts": self._upscaling_artifact_severity(gray),
            "pixelization": self._pixelization_severity(gray),
            "noise": self._noise_severity(gray),
        }
        criteria = {
            name: self._criterion(name, severity)
            for name, severity in severities.items()
        }
        score = self._quality_score(severities)

        return PhotoQualityAssessment(
            score=self._round(score),
            label=self._quality_label(score),
            is_acceptable=score >= self.min_acceptable_score,
            width=rgb.width,
            height=rgb.height,
            upscaling_artifacts=criteria["upscaling_artifacts"],
            pixelization=criteria["pixelization"],
            noise=criteria["noise"],
            metrics={
                "upscaling_detail_loss": criteria["upscaling_artifacts"].severity,
                "pixelization_blockiness": criteria["pixelization"].severity,
                "noise_residual_mad": criteria["noise"].severity,
            },
            reasons=self._reasons(severities),
        )

    def evaluate(self, image: ImageInput) -> PhotoQualityAssessment:
        return self.assess(image)

    def _quality_score(self, severities: Mapping[str, float]) -> float:
        total_weight = sum(self.weights.values())
        if total_weight <= 0:
            return 1.0

        weighted_badness = sum(
            self.weights.get(name, 0.0) * severity
            for name, severity in severities.items()
        )

        return self._clamp(1.0 - weighted_badness / total_weight)

    def _criterion(self, name: str, severity: float) -> PhotoQualityCriterion:
        return PhotoQualityCriterion(
            name=name,
            severity=self._round(severity),
            score=self._round(1.0 - severity),
            label=self._severity_label(severity),
        )

    @classmethod
    def _upscaling_artifact_severity(cls, gray: np.ndarray) -> float:
        if min(gray.shape) < 4:
            return 0.0

        detail_loss = cls._normalize_reverse(
            cls._laplacian_abs_mean(gray),
            low=0.0025,
            high=0.0065,
        )
        easy_to_reconstruct = cls._normalize_reverse(
            cls._resize_residual(gray),
            low=0.0015,
            high=0.006,
        )

        return cls._clamp(detail_loss * easy_to_reconstruct)

    @classmethod
    def _pixelization_severity(cls, gray: np.ndarray) -> float:
        if min(gray.shape) < 8:
            return 0.0

        horizontal = np.abs(np.diff(gray, axis=0))
        vertical = np.abs(np.diff(gray, axis=1))
        blockiness = max(
            cls._block_boundary_ratio(horizontal, gray.shape[0], axis=0, block_size=4),
            cls._block_boundary_ratio(horizontal, gray.shape[0], axis=0, block_size=8),
            cls._block_boundary_ratio(vertical, gray.shape[1], axis=1, block_size=4),
            cls._block_boundary_ratio(vertical, gray.shape[1], axis=1, block_size=8),
        )

        return cls._normalize(blockiness, low=1.15, high=2.4)

    @classmethod
    def _noise_severity(cls, gray: np.ndarray) -> float:
        if min(gray.shape) < 3:
            return 0.0

        residual = gray - cls._box_blur(gray)
        gy, gx = np.gradient(gray)
        gradient = np.sqrt(gx * gx + gy * gy)
        flat_mask = gradient <= np.percentile(gradient, 60)
        sample = residual[flat_mask] if np.any(flat_mask) else residual.ravel()
        median = float(np.median(sample))
        mad = float(np.median(np.abs(sample - median)) * 1.4826)

        return cls._normalize(mad, low=0.01, high=0.07)

    @staticmethod
    def _load_rgb_image(image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert("RGB")

        with Image.open(image) as opened_image:
            return ImageOps.exif_transpose(opened_image).convert("RGB").copy()

    @staticmethod
    def _to_luminance(image: Image.Image) -> np.ndarray:
        max_side = 512
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )

        return np.asarray(ImageOps.grayscale(image), dtype=np.float32) / 255.0

    @staticmethod
    def _resize_residual(gray: np.ndarray) -> float:
        height, width = gray.shape
        source = Image.fromarray(np.clip(gray * 255.0, 0, 255).astype(np.uint8))
        downscaled = source.resize(
            (max(1, width // 2), max(1, height // 2)),
            Image.Resampling.BICUBIC,
        )
        restored = downscaled.resize((width, height), Image.Resampling.BICUBIC)
        restored_array = np.asarray(restored, dtype=np.float32) / 255.0

        return float(np.mean(np.abs(gray - restored_array)))

    @staticmethod
    def _laplacian_abs_mean(gray: np.ndarray) -> float:
        padded = np.pad(gray, pad_width=1, mode="edge")
        laplacian = (
            -4.0 * padded[1:-1, 1:-1]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )

        return float(np.mean(np.abs(laplacian)))

    @staticmethod
    def _block_boundary_ratio(
        differences: np.ndarray,
        axis_length: int,
        axis: int,
        block_size: int,
    ) -> float:
        positions = np.arange(1, axis_length)
        boundary_mask = positions % block_size == 0
        if not np.any(boundary_mask) or not np.any(~boundary_mask):
            return 1.0

        if axis == 0:
            boundary_values = differences[boundary_mask, :]
            interior_values = differences[~boundary_mask, :]
        else:
            boundary_values = differences[:, boundary_mask]
            interior_values = differences[:, ~boundary_mask]

        return float(np.mean(boundary_values) / (np.mean(interior_values) + 1e-6))

    @staticmethod
    def _box_blur(gray: np.ndarray) -> np.ndarray:
        padded = np.pad(gray, pad_width=1, mode="edge")

        return (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0

    @staticmethod
    def _reasons(severities: Mapping[str, float]) -> list[str]:
        messages = {
            "upscaling_artifacts": "visible upscaling or interpolation artifacts",
            "pixelization": "visible pixelization or block boundaries",
            "noise": "visible sensor/compression noise",
        }
        reasons = [
            messages[name]
            for name, severity in severities.items()
            if severity >= 0.45
        ]

        return reasons or ["no dominant quality artifact detected"]

    @staticmethod
    def _quality_label(score: float) -> str:
        if score >= 0.82:
            return "good"
        if score >= 0.65:
            return "acceptable"
        if score >= 0.45:
            return "poor"

        return "bad"

    @staticmethod
    def _severity_label(severity: float) -> str:
        if severity < 0.2:
            return "low"
        if severity < 0.5:
            return "medium"
        if severity < 0.75:
            return "high"

        return "critical"

    @classmethod
    def _normalize(cls, value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0

        return cls._clamp((value - low) / (high - low))

    @classmethod
    def _normalize_reverse(cls, value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0

        return cls._clamp((high - value) / (high - low))

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return min(high, max(low, float(value)))

    @staticmethod
    def _round(value: float) -> float:
        return round(float(value), 4)
