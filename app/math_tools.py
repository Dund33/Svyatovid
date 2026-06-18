from typing import Iterable

import numpy as np
from sklearn.metrics import pairwise_distances


def largest_cosine_similarity_component(
    vectors: Iterable[np.ndarray],
    similarity_threshold: float = 0.6,
) -> list[np.ndarray]:
    """Return arrays from the largest connected component by cosine similarity."""
    items, matrix = _to_matrix(vectors)

    if not items:
        return []

    normalized = _normalize_rows(matrix)
    similarities = normalized @ normalized.T
    adjacency = similarities >= similarity_threshold
    components = _connected_components(adjacency)

    if not components:
        return []

    largest_component = max(components, key=len)

    return [items[index] for index in largest_component]


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


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero vector.")

    return matrix / norms


def _connected_components(adjacency: np.ndarray) -> list[list[int]]:
    visited = set()
    components = []

    for start_index in range(adjacency.shape[0]):
        if start_index in visited:
            continue

        component = []
        stack = [start_index]
        visited.add(start_index)

        while stack:
            index = stack.pop()
            component.append(index)

            neighbors = np.flatnonzero(adjacency[index])
            for neighbor in neighbors:
                neighbor = int(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(component)

    return components
