from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random


TINYFACE_DATASET_DIR = Path("tinyface/tinyface")
OUTPUT_DIR = Path("data/generated_tinyface_lowres")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGE_SIZE = 250


def generate_tinyface_lowres_dataset(
    dataset_dir: str | Path = TINYFACE_DATASET_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    min_register: int = 1,
    min_test: int = 1,
    num_register: int | None = 3,
    num_test: int | None = 3,
    max_people: int | None = 10,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int | None = None,
) -> Path:
    _validate_positive("min_register", min_register)
    _validate_positive("min_test", min_test)
    _validate_positive("image_size", image_size)
    _validate_optional_positive("num_register", num_register)
    _validate_optional_positive("num_test", num_test)
    _validate_optional_positive("max_people", max_people)

    tinyface_path = Path(dataset_dir)
    output_path = Path(output_dir)
    testing_set_dir = tinyface_path / "Testing_Set"
    rng = random.Random(seed)

    gallery_by_id = _load_tinyface_pairs(
        mat_path=testing_set_dir / "gallery_match_img_ID_pairs.mat",
        image_key="gallery_set",
        id_key="gallery_ids",
        image_dir=testing_set_dir / "Gallery_Match",
    )
    probe_by_id = _load_tinyface_pairs(
        mat_path=testing_set_dir / "probe_img_ID_pairs.mat",
        image_key="probe_set",
        id_key="probe_ids",
        image_dir=testing_set_dir / "Probe",
    )
    person_ids = _eligible_person_ids(
        gallery_by_id=gallery_by_id,
        probe_by_id=probe_by_id,
        min_register=min_register,
        min_test=min_test,
    )
    if not person_ids:
        raise ValueError("No TinyFace identities have enough gallery and probe images.")

    rng.shuffle(person_ids)
    if max_people is not None:
        person_ids = person_ids[:max_people]

    for person_id in sorted(person_ids):
        register_images = _select_images(
            gallery_by_id[person_id],
            limit=num_register,
            rng=rng,
        )
        test_images = _select_images(
            probe_by_id[person_id],
            limit=num_test,
            rng=rng,
        )
        _copy_split_images(
            image_paths=register_images,
            output_dir=output_path / f"id_{person_id}" / "register",
            image_size=image_size,
        )
        _copy_split_images(
            image_paths=test_images,
            output_dir=output_path / f"id_{person_id}" / "test",
            image_size=image_size,
        )

    return output_path


def _load_tinyface_pairs(
    mat_path: Path,
    image_key: str,
    id_key: str,
    image_dir: Path,
) -> dict[int, list[Path]]:
    _validate_path_exists(mat_path)
    _validate_path_exists(image_dir)

    try:
        from scipy.io import loadmat
    except ImportError as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "scipy is required to read TinyFace .mat metadata. "
            "Install project requirements before running this script."
        ) from error

    metadata = loadmat(mat_path)
    image_names = metadata[image_key].reshape(-1)
    identity_ids = metadata[id_key].reshape(-1)
    images_by_id: dict[int, list[Path]] = defaultdict(list)

    for image_name_value, identity_id_value in zip(image_names, identity_ids):
        image_name = _mat_string(image_name_value)
        identity_id = int(identity_id_value)
        image_path = image_dir / image_name

        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images_by_id[identity_id].append(image_path)

    return dict(images_by_id)


def _mat_string(value) -> str:
    while hasattr(value, "shape") and value.shape:
        value = value.reshape(-1)[0]

    return str(value)


def _eligible_person_ids(
    gallery_by_id: dict[int, list[Path]],
    probe_by_id: dict[int, list[Path]],
    min_register: int,
    min_test: int,
) -> list[int]:
    return sorted(
        person_id
        for person_id in gallery_by_id.keys() & probe_by_id.keys()
        if len(gallery_by_id[person_id]) >= min_register
        and len(probe_by_id[person_id]) >= min_test
    )


def _select_images(
    image_paths: list[Path],
    limit: int | None,
    rng: random.Random,
) -> list[Path]:
    if limit is None or limit >= len(image_paths):
        return sorted(image_paths)

    return sorted(rng.sample(image_paths, limit))


def _copy_split_images(
    image_paths: list[Path],
    output_dir: Path,
    image_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        _save_upscaled_image(image_path, output_dir / image_path.name, image_size)


def _save_upscaled_image(
    image_path: Path,
    output_path: Path,
    image_size: int,
) -> None:
    from PIL import Image, ImageOps

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        image.save(output_path, quality=95)


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")


def _validate_optional_positive(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be at least 1.")


def _validate_path_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _parse_optional_count(value: str) -> int | None:
    count = int(value)

    return None if count == 0 else count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(TINYFACE_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--min-register", type=int, default=1)
    parser.add_argument("--min-test", type=int, default=1)
    parser.add_argument("--num-register", type=_parse_optional_count, default=3)
    parser.add_argument("--num-test", type=_parse_optional_count, default=3)
    parser.add_argument("--max-people", type=_parse_optional_count, default=10)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_tinyface_lowres_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        min_register=args.min_register,
        min_test=args.min_test,
        num_register=args.num_register,
        num_test=args.num_test,
        max_people=args.max_people,
        image_size=args.image_size,
        seed=args.seed,
    )
