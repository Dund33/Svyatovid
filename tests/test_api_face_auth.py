from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from PIL import Image, ImageOps


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset" / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_API_BASE_URL = "http://localhost:5000"
TEST_PERSON_COUNT = 10
MAX_FRR = 0.15
MAX_FAR = 0.15
UPSCALED_ARTIFACT_SCALE = 0.5
MAX_UPSCALED_ARTIFACT_FRR = 0.25
MAX_UPSCALED_ARTIFACT_FAR = MAX_FAR


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    body: str


@dataclass(frozen=True)
class PersonLoginResult:
    person: str
    successful_logins: int
    total_logins: int

    @property
    def false_rejections(self) -> int:
        return self.total_logins - self.successful_logins

    @property
    def frr(self) -> float:
        return self.false_rejections / self.total_logins


@dataclass(frozen=True)
class PersonImpostorResult:
    registered_person: str
    impostor_person: str
    false_acceptances: int
    total_attempts: int

    @property
    def far(self) -> float:
        return self.false_acceptances / self.total_attempts


class FaceAuthEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ping_response = _request("GET", "/ping")
        except URLError as error:  # pragma: no cover - depends on local API
            raise unittest.SkipTest(
                f"API is not available at {_api_base_url()}: {error}"
            ) from error

        if ping_response.status_code != 200:
            raise unittest.SkipTest(
                f"API is not healthy at {_api_base_url()}: "
                f"{ping_response.status_code} {ping_response.body}"
            )

        clear_response = _request("POST", "/clear")
        if clear_response.status_code != 200:
            raise AssertionError(
                f"Could not clear API data before test: "
                f"{clear_response.status_code} {clear_response.body}"
            )

    @classmethod
    def tearDownClass(cls):
        try:
            _request("POST", "/clear")
        except URLError:
            pass

    def test_register_profiles_and_verify_frr(self):
        person_dirs = _find_person_dirs()
        results = []

        for person_dir in person_dirs:
            with self.subTest(person=person_dir.name):
                _clear_api_data()
                results.append(self._test_person_register_and_login(person_dir))

            _clear_api_data()

        total_logins = sum(result.total_logins for result in results)
        total_false_rejections = sum(result.false_rejections for result in results)
        frr = total_false_rejections / total_logins
        max_frr = _max_frr()

        print(
            f"Tested {len(person_dirs)} people: "
            f"FRR={frr:.4f} false_rejections={total_false_rejections} "
            f"total_logins={total_logins} max_frr={max_frr:.4f} "
            f"results={results}"
        )

        self.assertLessEqual(
            frr,
            max_frr,
            (
                f"FRR {frr:.4f} is greater than max allowed FRR {max_frr:.4f}. "
                f"False rejections: {total_false_rejections}/{total_logins}. "
                f"Results: {results}"
            ),
        )

    def test_register_profiles_and_verify_far(self):
        person_dirs = _find_person_dirs()
        results = []

        for index, person_dir in enumerate(person_dirs):
            impostor_dir = person_dirs[(index + 1) % len(person_dirs)]
            with self.subTest(
                registered_person=person_dir.name,
                impostor_person=impostor_dir.name,
            ):
                _clear_api_data()
                results.append(self._test_person_rejects_impostor(person_dir, impostor_dir))

            _clear_api_data()

        total_attempts = sum(result.total_attempts for result in results)
        total_false_acceptances = sum(result.false_acceptances for result in results)
        far = total_false_acceptances / total_attempts
        max_far = _max_far()

        print(
            f"Tested {len(person_dirs)} impostor pairs: "
            f"FAR={far:.4f} false_acceptances={total_false_acceptances} "
            f"total_attempts={total_attempts} max_far={max_far:.4f} "
            f"results={results}"
        )

        self.assertLessEqual(
            far,
            max_far,
            (
                f"FAR {far:.4f} is greater than max allowed FAR {max_far:.4f}. "
                f"False acceptances: {total_false_acceptances}/{total_attempts}. "
                f"Results: {results}"
            ),
        )

    def test_register_profiles_and_verify_frr_with_upscaling_artifacts(self):
        person_dirs = _find_person_dirs()
        results = []

        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir)

            for person_dir in person_dirs:
                with self.subTest(person=person_dir.name):
                    _clear_api_data()
                    results.append(
                        self._test_person_register_and_login_with_upscaling_artifacts(
                            person_dir=person_dir,
                            artifact_dir=artifact_root / person_dir.name,
                        )
                    )

                _clear_api_data()

        total_logins = sum(result.total_logins for result in results)
        total_false_rejections = sum(result.false_rejections for result in results)
        frr = total_false_rejections / total_logins
        max_frr = _max_upscaled_artifact_frr()

        print(
            f"Tested {len(person_dirs)} people with upscaling artifacts: "
            f"FRR={frr:.4f} false_rejections={total_false_rejections} "
            f"total_logins={total_logins} max_frr={max_frr:.4f} "
            f"results={results}"
        )

        self.assertLessEqual(
            frr,
            max_frr,
            (
                f"FRR {frr:.4f} with upscaling artifacts is greater than "
                f"max allowed FRR {max_frr:.4f}. "
                f"False rejections: {total_false_rejections}/{total_logins}. "
                f"Results: {results}"
            ),
        )

    def test_register_profiles_and_verify_far_with_upscaling_artifacts(self):
        person_dirs = _find_person_dirs()
        results = []

        with TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir)

            for index, person_dir in enumerate(person_dirs):
                impostor_dir = person_dirs[(index + 1) % len(person_dirs)]
                with self.subTest(
                    registered_person=person_dir.name,
                    impostor_person=impostor_dir.name,
                ):
                    _clear_api_data()
                    results.append(
                        self._test_person_rejects_impostor_with_upscaling_artifacts(
                            person_dir=person_dir,
                            impostor_dir=impostor_dir,
                            artifact_dir=artifact_root
                            / f"{person_dir.name}_vs_{impostor_dir.name}",
                        )
                    )

                _clear_api_data()

        total_attempts = sum(result.total_attempts for result in results)
        total_false_acceptances = sum(result.false_acceptances for result in results)
        far = total_false_acceptances / total_attempts
        max_far = _max_upscaled_artifact_far()

        print(
            f"Tested {len(person_dirs)} impostor pairs with upscaling artifacts: "
            f"FAR={far:.4f} false_acceptances={total_false_acceptances} "
            f"total_attempts={total_attempts} max_far={max_far:.4f} "
            f"results={results}"
        )

        self.assertLessEqual(
            far,
            max_far,
            (
                f"FAR {far:.4f} with upscaling artifacts is greater than "
                f"max allowed FAR {max_far:.4f}. "
                f"False acceptances: {total_false_acceptances}/{total_attempts}. "
                f"Results: {results}"
            ),
        )

    def _test_person_register_and_login(self, person_dir: Path) -> PersonLoginResult:
        test_dir = person_dir / "test"
        test_images = _image_paths(test_dir)

        self._register_person(person_dir)
        self.assertTrue(test_images, f"No test images found in {test_dir}")

        successful_logins = 0
        unexpected_responses = []

        for image_path in test_images:
            response = _post_files(
                endpoint="/login",
                fields={},
                image_paths=[image_path],
            )

            if response.status_code == 200:
                successful_logins += 1
            elif response.status_code != 401:
                unexpected_responses.append(
                    (image_path.name, response.status_code, response.body)
                )

        print(
            f"{person_dir.name}: logged in with "
            f"{successful_logins}/{len(test_images)} images from {test_dir} "
            f"FRR={(len(test_images) - successful_logins) / len(test_images):.4f}"
        )

        self.assertEqual(unexpected_responses, [])

        return PersonLoginResult(
            person=person_dir.name,
            successful_logins=successful_logins,
            total_logins=len(test_images),
        )

    def _test_person_register_and_login_with_upscaling_artifacts(
        self,
        person_dir: Path,
        artifact_dir: Path,
    ) -> PersonLoginResult:
        register_dir = person_dir / "register"
        test_dir = person_dir / "test"
        register_images = _upscaled_artifact_images(
            _image_paths(register_dir),
            artifact_dir / "register",
        )
        test_images = _upscaled_artifact_images(
            _image_paths(test_dir),
            artifact_dir / "test",
        )

        self._register_person_with_images(person_dir.name, register_images)
        self.assertTrue(test_images, f"No test images found in {test_dir}")

        successful_logins = 0
        unexpected_responses = []

        for image_path in test_images:
            response = _post_files(
                endpoint="/login",
                fields={},
                image_paths=[image_path],
            )

            if response.status_code == 200:
                successful_logins += 1
            elif response.status_code != 401:
                unexpected_responses.append(
                    (image_path.name, response.status_code, response.body)
                )

        print(
            f"{person_dir.name} with upscaling artifacts: logged in with "
            f"{successful_logins}/{len(test_images)} images from {test_dir} "
            f"FRR={(len(test_images) - successful_logins) / len(test_images):.4f}"
        )

        self.assertEqual(unexpected_responses, [])

        return PersonLoginResult(
            person=person_dir.name,
            successful_logins=successful_logins,
            total_logins=len(test_images),
        )

    def _test_person_rejects_impostor(
        self,
        person_dir: Path,
        impostor_dir: Path,
    ) -> PersonImpostorResult:
        impostor_test_dir = impostor_dir / "test"
        impostor_test_images = _image_paths(impostor_test_dir)

        self._register_person(person_dir)
        self.assertTrue(
            impostor_test_images,
            f"No impostor test images found in {impostor_test_dir}",
        )

        false_acceptances = 0
        unexpected_responses = []

        for image_path in impostor_test_images:
            response = _post_files(
                endpoint="/login",
                fields={},
                image_paths=[image_path],
            )

            if response.status_code == 200:
                false_acceptances += 1
            elif response.status_code != 401:
                unexpected_responses.append(
                    (image_path.name, response.status_code, response.body)
                )

        print(
            f"{person_dir.name} vs {impostor_dir.name}: false accepted "
            f"{false_acceptances}/{len(impostor_test_images)} images from "
            f"{impostor_test_dir} FAR={false_acceptances / len(impostor_test_images):.4f}"
        )

        self.assertEqual(unexpected_responses, [])

        return PersonImpostorResult(
            registered_person=person_dir.name,
            impostor_person=impostor_dir.name,
            false_acceptances=false_acceptances,
            total_attempts=len(impostor_test_images),
        )

    def _test_person_rejects_impostor_with_upscaling_artifacts(
        self,
        person_dir: Path,
        impostor_dir: Path,
        artifact_dir: Path,
    ) -> PersonImpostorResult:
        register_dir = person_dir / "register"
        impostor_test_dir = impostor_dir / "test"
        register_images = _upscaled_artifact_images(
            _image_paths(register_dir),
            artifact_dir / "register" / person_dir.name,
        )
        impostor_test_images = _upscaled_artifact_images(
            _image_paths(impostor_test_dir),
            artifact_dir / "test" / impostor_dir.name,
        )

        self._register_person_with_images(person_dir.name, register_images)
        self.assertTrue(
            impostor_test_images,
            f"No impostor test images found in {impostor_test_dir}",
        )

        false_acceptances = 0
        unexpected_responses = []

        for image_path in impostor_test_images:
            response = _post_files(
                endpoint="/login",
                fields={},
                image_paths=[image_path],
            )

            if response.status_code == 200:
                false_acceptances += 1
            elif response.status_code != 401:
                unexpected_responses.append(
                    (image_path.name, response.status_code, response.body)
                )

        print(
            f"{person_dir.name} vs {impostor_dir.name} with upscaling artifacts: "
            f"false accepted {false_acceptances}/{len(impostor_test_images)} "
            f"images from {impostor_test_dir} "
            f"FAR={false_acceptances / len(impostor_test_images):.4f}"
        )

        self.assertEqual(unexpected_responses, [])

        return PersonImpostorResult(
            registered_person=person_dir.name,
            impostor_person=impostor_dir.name,
            false_acceptances=false_acceptances,
            total_attempts=len(impostor_test_images),
        )

    def _register_person(self, person_dir: Path) -> None:
        register_dir = person_dir / "register"
        register_images = _image_paths(register_dir)

        self._register_person_with_images(person_dir.name, register_images)

    def _register_person_with_images(
        self,
        person_name: str,
        register_images: list[Path],
    ) -> None:
        self.assertTrue(register_images, "No register images found.")

        response = _post_files(
            endpoint="/register",
            fields={
                "username": f"test_{person_name}_{uuid4().hex[:8]}",
                "password": "test-password",
            },
            image_paths=register_images,
        )

        self.assertEqual(response.status_code, 201, response.body)


def _request(
    method: str,
    endpoint: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> HttpResponse:
    request = Request(
        _api_url(endpoint),
        data=body,
        headers=headers or {},
        method=method,
    )

    try:
        with urlopen(request, timeout=_request_timeout()) as response:
            return HttpResponse(
                status_code=response.status,
                body=response.read().decode("utf-8", errors="replace"),
            )
    except HTTPError as error:
        return HttpResponse(
            status_code=error.code,
            body=error.read().decode("utf-8", errors="replace"),
        )


def _clear_api_data() -> None:
    clear_response = _request("POST", "/clear")
    if clear_response.status_code != 200:
        raise AssertionError(
            f"Could not clear API data: "
            f"{clear_response.status_code} {clear_response.body}"
        )


def _find_person_dirs() -> list[Path]:
    dataset_dir = Path(os.getenv("SVYATOVID_TEST_DATASET_DIR", DATASET_DIR))
    person_name = os.getenv("SVYATOVID_TEST_PERSON")

    if person_name:
        candidates = [
            dataset_dir / person_name,
            dataset_dir / "generated_lfw_composites" / person_name,
        ]
    else:
        candidates = sorted(path.parent for path in dataset_dir.rglob("register"))

    person_dirs = []
    for person_dir in candidates:
        if (
            person_dir.is_dir()
            and _image_paths(person_dir / "register")
            and _image_paths(person_dir / "test")
        ):
            person_dirs.append(person_dir)

    if not person_dirs:
        raise unittest.SkipTest("No dataset person found with register and test images.")

    if person_name:
        return person_dirs[:1]

    if len(person_dirs) < TEST_PERSON_COUNT:
        raise unittest.SkipTest(
            f"Found only {len(person_dirs)} people with register and test images, "
            f"but {TEST_PERSON_COUNT} are required."
        )

    return person_dirs[:TEST_PERSON_COUNT]


def _image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _upscaled_artifact_images(image_paths: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = []
    for index, image_path in enumerate(image_paths):
        output_path = output_dir / f"{index:03d}_{image_path.stem}_upscaled.png"
        _save_upscaled_artifact_image(image_path, output_path)
        artifact_paths.append(output_path)

    return artifact_paths


def _save_upscaled_artifact_image(image_path: Path, output_path: Path) -> None:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        downscaled_size = (
            max(1, int(round(image.width * UPSCALED_ARTIFACT_SCALE))),
            max(1, int(round(image.height * UPSCALED_ARTIFACT_SCALE))),
        )
        upscaled = image.resize(
            downscaled_size,
            Image.Resampling.BICUBIC,
        ).resize(
            image.size,
            Image.Resampling.BICUBIC,
        )
        upscaled.save(output_path, format="PNG")


def _post_files(
    endpoint: str,
    fields: dict[str, str],
    image_paths: list[Path],
) -> HttpResponse:
    body, headers = _multipart_body(fields=fields, image_paths=image_paths)

    return _request("POST", endpoint, body=body, headers=headers)


def _multipart_body(
    fields: dict[str, str],
    image_paths: list[Path],
) -> tuple[bytes, dict[str, str]]:
    boundary = f"----SvyatovidTestBoundary{uuid4().hex}"
    chunks = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}".encode(),
                f'Content-Disposition: form-data; name="{_escape_header(name)}"'.encode(),
                b"",
                value.encode(),
            ]
        )

    for image_path in image_paths:
        chunks.extend(
            [
                f"--{boundary}".encode(),
                (
                    "Content-Disposition: form-data; "
                    f'name="images"; filename="{_escape_header(image_path.name)}"'
                ).encode(),
                b"Content-Type: application/octet-stream",
                b"",
                image_path.read_bytes(),
            ]
        )

    chunks.extend([f"--{boundary}--".encode(), b""])

    return b"\r\n".join(chunks), {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }


def _api_url(endpoint: str) -> str:
    return f"{_api_base_url()}/{endpoint.lstrip('/')}"


def _api_base_url() -> str:
    return os.getenv("SVYATOVID_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def _request_timeout() -> float:
    return float(os.getenv("SVYATOVID_TEST_TIMEOUT", "300"))


def _max_frr() -> float:
    return MAX_FRR


def _max_far() -> float:
    return MAX_FAR


def _max_upscaled_artifact_frr() -> float:
    return MAX_UPSCALED_ARTIFACT_FRR


def _max_upscaled_artifact_far() -> float:
    return MAX_UPSCALED_ARTIFACT_FAR


def _escape_header(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
