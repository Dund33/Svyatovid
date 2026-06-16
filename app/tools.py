from pathlib import Path
from typing import Iterable

import numpy as np

from ai_tools import face_embeddings_from_images
from math_tools import largest_dbscan_cluster, medoid


ImagePath = str | Path


def create_user_embedding_from_images(image_paths: Iterable[ImagePath]) -> np.ndarray:
    embeddings = face_embeddings_from_images(image_paths)

    if not embeddings:
        raise ValueError("No face embeddings found.")

    cluster = largest_dbscan_cluster(embeddings)

    if not cluster:
        raise ValueError("No face cluster found.")

    return medoid(cluster)
