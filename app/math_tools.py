from typing import Iterable

import numpy as np


def largest_cosine_similarity_component(
    vectors: Iterable[np.ndarray],
    similarity_threshold: float = 0.6,
) -> list[np.ndarray]:
    """Return arrays from the largest connected component by cosine similarity."""
    items = list(vectors)
    indices = largest_cosine_similarity_component_indices(
        items,
        similarity_threshold=similarity_threshold,
    )

    return [items[index] for index in indices]


def largest_cosine_similarity_component_indices(
    vectors: Iterable[np.ndarray],
    similarity_threshold: float = 0.6,
) -> list[int]:
    """Return indices from the largest connected component by cosine similarity."""
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

    return largest_component


def medoid(vectors: Iterable[np.ndarray], metric: str = "euclidean") -> np.ndarray:
    """Return the input array with the smallest total distance to all others."""
    items, matrix = _to_matrix(vectors)

    if not items:
        raise ValueError("Cannot compute medoid of an empty list.")

    distances = _pairwise_distances(matrix, metric=metric)
    medoid_index = int(np.argmin(distances.sum(axis=1)))

    return items[medoid_index]


def weighted_medoid(
    vectors: Iterable[np.ndarray],
    weights: Iterable[float],
    metric: str = "euclidean",
) -> np.ndarray:
    """Return the input array with the smallest weighted total distance."""
    items, matrix = _to_matrix(vectors)

    if not items:
        raise ValueError("Cannot compute weighted medoid of an empty list.")

    weight_array = _to_weight_array(weights, len(items))
    distances = _pairwise_distances(matrix, metric=metric)
    medoid_index = int(np.argmin(distances @ weight_array))

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


def _to_weight_array(weights: Iterable[float], expected_length: int) -> np.ndarray:
    weight_array = np.asarray(list(weights), dtype=float)

    if weight_array.shape != (expected_length,):
        raise ValueError("Weights must have the same length as vectors.")

    if not np.all(np.isfinite(weight_array)):
        raise ValueError("Weights must be finite numbers.")

    if np.any(weight_array < 0):
        raise ValueError("Weights cannot be negative.")

    if np.sum(weight_array) == 0:
        return np.ones(expected_length, dtype=float)

    return weight_array


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero vector.")

    return matrix / norms


def _pairwise_distances(matrix: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        differences = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]

        return np.sqrt(np.sum(differences * differences, axis=2))

    if metric == "cosine":
        normalized = _normalize_rows(matrix)
        distances = 1.0 - normalized @ normalized.T

        return np.clip(distances, 0.0, 2.0)

    from sklearn.metrics import pairwise_distances

    return pairwise_distances(matrix, metric=metric)


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
