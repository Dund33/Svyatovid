from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


LFW_DATASET = "jessicali9530/lfw-dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def generate_lfw_composite_dataset(
    output_dir: str | Path = "data/generated_lfw_composites",
    image_size: int = 750,
    face_size: int = 250,
    min_imgs: int = 3,
    num_register: int = 3,
    seed: int | None = None,
) -> Path:
    _validate_sizes(image_size=image_size, face_size=face_size)
    _validate_min_imgs(min_imgs)
    _validate_num_register(num_register, min_imgs)

    dataset_path = load_lfw_dataset()
    people_images = _filter_people_by_image_count(
        people_images=_load_people_images(dataset_path),
        min_imgs=min_imgs,
    )
    output_path = Path(output_dir)
    rng = random.Random(seed)

    _validate_people_images(people_images)
    _create_composite_dataset(
        people_images=people_images,
        output_dir=output_path,
        image_size=image_size,
        face_size=face_size,
        num_register=num_register,
        rng=rng,
    )

    return output_path


def load_lfw_dataset() -> Path:
    import kagglehub

    path = kagglehub.dataset_download(LFW_DATASET)

    return Path(path)


def _validate_sizes(image_size: int, face_size: int) -> None:
    if image_size <= 0:
        raise ValueError("image_size must be greater than 0.")

    if face_size <= 0:
        raise ValueError("face_size must be greater than 0.")

    if face_size > image_size:
        raise ValueError("face_size cannot be greater than image_size.")


def _validate_min_imgs(min_imgs: int) -> None:
    if min_imgs < 1:
        raise ValueError("min_imgs must be at least 1.")


def _validate_num_register(num_register: int, min_imgs: int) -> None:
    if num_register < 1:
        raise ValueError("num_register must be at least 1.")

    if num_register > min_imgs:
        raise ValueError("num_register cannot be greater than min_imgs.")


def _load_people_images(dataset_path: Path) -> dict[str, list[Path]]:
    people_images: dict[str, list[Path]] = {}

    for image_path in _find_image_paths(dataset_path):
        person_name = image_path.parent.name
        people_images.setdefault(person_name, []).append(image_path)

    return people_images


def _filter_people_by_image_count(
    people_images: dict[str, list[Path]],
    min_imgs: int,
) -> dict[str, list[Path]]:
    return {
        person_name: image_paths
        for person_name, image_paths in people_images.items()
        if len(image_paths) >= min_imgs
    }


def _find_image_paths(dataset_path: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _validate_people_images(people_images: dict[str, list[Path]]) -> None:
    if len(people_images) < 2:
        raise ValueError("Dataset must contain images for at least two people.")

    for person_name, image_paths in people_images.items():
        if not image_paths:
            raise ValueError(f"No images found for {person_name}.")


def _create_composite_dataset(
    people_images: dict[str, list[Path]],
    output_dir: Path,
    image_size: int,
    face_size: int,
    num_register: int,
    rng: random.Random,
) -> None:
    for person_name, image_paths in people_images.items():
        register_images = set(rng.sample(image_paths, num_register))

        for image_path in image_paths:
            other_face_path = _choose_other_person_face(
                people_images=people_images,
                current_person=person_name,
                rng=rng,
            )
            composite = _create_composite_image(
                base_image_path=image_path,
                face_image_path=other_face_path,
                image_size=image_size,
                face_size=face_size,
                rng=rng,
            )
            split_name = "register" if image_path in register_images else "test"
            split_dir = output_dir / person_name / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            composite.save(split_dir / f"{image_path.stem}.jpg", quality=95)


def _choose_other_person_face(
    people_images: dict[str, list[Path]],
    current_person: str,
    rng: random.Random,
) -> Path:
    other_people = [person for person in people_images if person != current_person]
    other_person = rng.choice(other_people)

    return rng.choice(people_images[other_person])


def _create_composite_image(
    base_image_path: Path,
    face_image_path: Path,
    image_size: int,
    face_size: int,
    rng: random.Random,
) -> Image.Image:
    from PIL import Image

    composite = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
    base_image = _load_square_image(base_image_path, face_size)
    face_image = _load_square_image(face_image_path, face_size)
    base_position, face_position = _random_image_positions(
        image_size=image_size,
        face_size=face_size,
        rng=rng,
    )

    composite.paste(base_image, base_position)
    composite.paste(face_image, face_position)

    return composite


def _load_square_image(image_path: Path, size: int) -> Image.Image:
    from PIL import Image, ImageOps

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")

        return ImageOps.fit(image, (size, size), Image.Resampling.LANCZOS)


def _random_image_positions(
    image_size: int,
    face_size: int,
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int]]:
    first_position = _random_image_position(
        image_size=image_size,
        face_size=face_size,
        rng=rng,
    )
    second_position = _random_non_overlapping_image_position(
        existing_position=first_position,
        image_size=image_size,
        face_size=face_size,
        rng=rng,
    )

    return first_position, second_position


def _random_non_overlapping_image_position(
    existing_position: tuple[int, int],
    image_size: int,
    face_size: int,
    rng: random.Random,
    attempts: int = 100,
) -> tuple[int, int]:
    for _ in range(attempts):
        candidate = _random_image_position(
            image_size=image_size,
            face_size=face_size,
            rng=rng,
        )

        if not _rectangles_overlap(
            first_position=existing_position,
            second_position=candidate,
            size=face_size,
        ):
            return candidate

    return _random_image_position(
        image_size=image_size,
        face_size=face_size,
        rng=rng,
    )


def _random_image_position(
    image_size: int,
    face_size: int,
    rng: random.Random,
) -> tuple[int, int]:
    max_coordinate = image_size - face_size

    return rng.randint(0, max_coordinate), rng.randint(0, max_coordinate)


def _rectangles_overlap(
    first_position: tuple[int, int],
    second_position: tuple[int, int],
    size: int,
) -> bool:
    first_x, first_y = first_position
    second_x, second_y = second_position

    return (
        first_x < second_x + size
        and first_x + size > second_x
        and first_y < second_y + size
        and first_y + size > second_y
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/generated_lfw_composites")
    parser.add_argument("--image-size", type=int, default=750)
    parser.add_argument("--face-size", type=int, default=250)
    parser.add_argument("--min-imgs", type=int, default=3)
    parser.add_argument("--num-register", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_lfw_composite_dataset(
        output_dir=args.output_dir,
        image_size=args.image_size,
        face_size=args.face_size,
        min_imgs=args.min_imgs,
        num_register=args.num_register,
        seed=args.seed,
    )
