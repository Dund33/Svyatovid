import os
from pathlib import Path
from typing import Iterable

import numpy as np

from ai_tools import face_embeddings_from_images
from math_tools import largest_cosine_similarity_component, medoid


ImagePath = str | Path


def create_user_embedding_from_images(image_paths: Iterable[ImagePath]) -> np.ndarray:
    image_paths = list(image_paths)
    embeddings = face_embeddings_from_images(image_paths)
    print(
        f"[api-debug] register embeddings found "
        f"embeddings={len(embeddings)} images={len(image_paths)}",
        flush=True,
    )
    if not embeddings:
        raise ValueError("No face embeddings found.")

    similarity_threshold = _face_cluster_similarity_threshold()
    cluster = largest_cosine_similarity_component(
        embeddings,
        similarity_threshold=similarity_threshold,
    )
    print(
        f"[api-debug] register largest component "
        f"size={len(cluster)} embeddings={len(embeddings)} "
        f"threshold={similarity_threshold}",
        flush=True,
    )

    if not cluster:
        raise ValueError("No face cluster found.")

    return medoid(cluster, metric="cosine")


def _face_cluster_similarity_threshold() -> float:
    return float(os.getenv("FACE_CLUSTER_SIMILARITY_THRESHOLD", "0.6"))
