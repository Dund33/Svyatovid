from collections import Counter
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances


def largest_dbscan_cluster(
    vectors: Iterable[np.ndarray],
    eps: float = 0.5,
    min_samples: int = 2,
    metric: str = "euclidean",
    **dbscan_kwargs: Any,
) -> list[np.ndarray]:
    """Return arrays belonging to the largest non-noise DBSCAN cluster."""
    items, matrix = _to_matrix(vectors)

    if not items:
        return []

    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        **dbscan_kwargs,
    ).fit_predict(matrix)

    label_counts = Counter(label for label in labels if label != -1)
    if not label_counts:
        return []

    largest_label, _ = label_counts.most_common(1)[0]

    return [item for item, label in zip(items, labels) if label == largest_label]


def medoid(vectors: Iterable[np.ndarray], metric: str = "euclidean") -> np.ndarray:
    """Return the input array with the smallest total distance to all others."""
    items, matrix = _to_matrix(vectors)

    if not items:
        raise ValueError("Cannot compute medoid of an empty list.")

    distances = pairwise_distances(matrix, metric=metric)
    medoid_index = int(np.argmin(distances.sum(axis=1)))

    return items[medoid_index]


def _to_matrix(vectors: Iterable[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    items = list(vectors)

    if not items:
        return [], np.empty((0, 0), dtype=float)

    flattened = []
    expected_shape = None

    for index, vector in enumerate(items):
        array = np.asarray(vector, dtype=float)
        if array.size == 0:
            raise ValueError(f"Vector at index {index} is empty.")

        flat = array.ravel()
        if expected_shape is None:
            expected_shape = flat.shape
        elif flat.shape != expected_shape:
            raise ValueError("All vectors must have the same flattened shape.")

        flattened.append(flat)

    return items, np.vstack(flattened)
