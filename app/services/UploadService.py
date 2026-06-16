from pathlib import Path
from uuid import uuid4


class UploadService:
    def __init__(self, upload_dir: str | Path = "/app/uploads"):
        self.upload_dir = Path(upload_dir)

    def save_files(self, files) -> list[str]:
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for file in self._iter_files(files):
            if not file or not file.filename:
                continue

            file_path = self.upload_dir / self._generate_filename(file.filename)
            file.save(str(file_path))
            saved_paths.append(str(file_path))

        return saved_paths

    @staticmethod
    def _iter_files(files):
        for field_name in files:
            yield from files.getlist(field_name)

    @staticmethod
    def _generate_filename(original_filename: str) -> str:
        suffix = Path(original_filename).suffix

        return f"{uuid4().hex}{suffix}"
