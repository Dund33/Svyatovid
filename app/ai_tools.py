from pathlib import Path
from typing import Iterable

import numpy as np
from deepface import DeepFace


ImageInput = str | Path | np.ndarray


def face_embeddings_from_image(
    image: ImageInput,
    model_name: str = "Facenet512",
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
) -> list[np.ndarray]:
    representations = _represent_faces(
        image=image,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    return [_embedding_to_array(representation) for representation in representations]


def face_embeddings_from_images(
    images: Iterable[ImageInput],
    model_name: str = "Facenet512",
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
) -> list[np.ndarray]:
    embeddings = []

    for image in images:
        embeddings.extend(
            face_embeddings_from_image(
                image=image,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                align=align,
            )
        )

    return embeddings


def _represent_faces(
    image: ImageInput,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
) -> list[dict]:
    result = DeepFace.represent(
        img_path=_prepare_image_input(image),
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    return _as_representation_list(result)


def _prepare_image_input(image: ImageInput) -> str | np.ndarray:
    if isinstance(image, Path):
        return str(image)

    return image


def _as_representation_list(result: dict | list[dict]) -> list[dict]:
    if isinstance(result, list):
        return result

    return [result]


def _embedding_to_array(representation: dict) -> np.ndarray:
    return np.asarray(representation["embedding"], dtype=np.float32)
