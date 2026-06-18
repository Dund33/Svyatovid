import os
from pathlib import Path
from typing import Iterable

import numpy as np

from ai_tools import face_embeddings_from_image
from math_tools import largest_cosine_similarity_component_indices, weighted_medoid
from services.PhotoQualityService import PhotoQualityService


ImagePath = str | Path


def create_user_embedding_from_images(
    image_paths: Iterable[ImagePath],
    photo_quality_service: PhotoQualityService | None = None,
) -> np.ndarray:
    image_paths = list(image_paths)
    embeddings, weights, quality_scores = _weighted_face_embeddings_from_images(
        image_paths,
        photo_quality_service=photo_quality_service or PhotoQualityService(),
    )
    print(
        f"[api-debug] register embeddings found "
        f"embeddings={len(embeddings)} images={len(image_paths)} "
        f"quality={_quality_summary(quality_scores)}",
        flush=True,
    )
    if not embeddings:
        raise ValueError("No face embeddings found.")

    similarity_threshold = _face_cluster_similarity_threshold()
    cluster_indices = largest_cosine_similarity_component_indices(
        embeddings,
        similarity_threshold=similarity_threshold,
    )
    cluster = [embeddings[index] for index in cluster_indices]
    cluster_weights = [weights[index] for index in cluster_indices]
    print(
        f"[api-debug] register largest component "
        f"size={len(cluster)} embeddings={len(embeddings)} "
        f"threshold={similarity_threshold} "
        f"quality_weights={_quality_summary(cluster_weights)}",
        flush=True,
    )

    if not cluster:
        raise ValueError("No face cluster found.")

    return weighted_medoid(cluster, weights=cluster_weights, metric="cosine")


def _weighted_face_embeddings_from_images(
    image_paths: list[ImagePath],
    photo_quality_service: PhotoQualityService,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    embeddings = []
    weights = []
    quality_scores = []

    for image_path in image_paths:
        quality_score = photo_quality_service.assess(image_path).score
        image_embeddings = face_embeddings_from_image(image_path)
        quality_scores.append(quality_score)
        embeddings.extend(image_embeddings)
        weights.extend([quality_score] * len(image_embeddings))

    return embeddings, weights, quality_scores


def _face_cluster_similarity_threshold() -> float:
    return float(os.getenv("FACE_CLUSTER_SIMILARITY_THRESHOLD", "0.6"))


def _quality_summary(scores: Iterable[float]) -> str:
    scores = list(scores)
    if not scores:
        return "none"

    return (
        f"min={min(scores):.4f} "
        f"avg={sum(scores) / len(scores):.4f} "
        f"max={max(scores):.4f}"
    )
