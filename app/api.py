import os

from flask import Flask, jsonify, request

from services.RedisStoreService import RedisStoreService
from services.UploadService import UploadService
from ai_tools import face_embeddings_from_images
from tools import create_user_embedding_from_images


app = Flask(__name__)

upload_service = UploadService(os.getenv("UPLOAD_FOLDER", "uploads"))
redis_store_service = RedisStoreService(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    embedding_dim=int(os.getenv("EMBEDDING_DIM", "512")),
)
face_match_threshold = float(os.getenv("FACE_MATCH_THRESHOLD", "0.3"))


@app.post("/register")
def register():
    username = request.form.get("username")
    password = request.form.get("password")
    _debug(
        "register request",
        username=username,
        file_fields=list(request.files.keys()),
        file_count=_uploaded_file_count(),
    )

    if not username or not password:
        _debug("register rejected", reason="missing username or password")
        return jsonify({"error": "username and password required"}), 400

    if not request.files:
        _debug("register rejected", reason="no files")
        return jsonify({"error": "at least one image file required"}), 400

    if redis_store_service.get_profile(username):
        _debug("register rejected", username=username, reason="user already exists")
        return jsonify({"error": "user already exists"}), 409

    file_paths = upload_service.save_files(request.files)
    _debug(
        "register files saved",
        count=len(file_paths),
        files=_file_names(file_paths),
    )

    if not file_paths:
        _debug("register rejected", reason="no files saved")
        return jsonify({"error": "at least one image file required"}), 400

    try:
        embedding = create_user_embedding_from_images(file_paths)
    except ValueError as error:
        _debug("register embedding error", error=error)
        return jsonify({"error": str(error)}), 400

    _debug("register embedding created", shape=getattr(embedding, "shape", None))
    redis_store_service.ensure_index()
    _debug("register redis index ensured")
    redis_store_service.create_profile(
        login=username,
        password=password,
        embedding=embedding,
    )
    _debug("register profile stored", username=username)
    _debug("register redis keys", count=_speaker_key_count())

    return jsonify({"message": "registered", "username": username}), 201


@app.post("/login")
def login():
    _debug(
        "login request",
        file_fields=list(request.files.keys()),
        file_count=_uploaded_file_count(),
    )

    if not request.files:
        _debug("login rejected", reason="no files")
        return jsonify({"error": "image file required"}), 400

    file_paths = upload_service.save_files(request.files)
    _debug("login files saved", count=len(file_paths), files=_file_names(file_paths))

    if not file_paths:
        _debug("login rejected", reason="no files saved")
        return jsonify({"error": "image file required"}), 400

    try:
        embeddings = face_embeddings_from_images(file_paths)
    except ValueError as error:
        _debug("login embedding error", error=error)
        return jsonify({"error": "unauthorized"}), 401

    _debug(
        "login embeddings created",
        count=len(embeddings),
        shapes=[getattr(embedding, "shape", None) for embedding in embeddings],
    )

    if not embeddings:
        _debug("login rejected", reason="no embeddings")
        return jsonify({"error": "unauthorized"}), 401

    redis_store_service.ensure_index()
    _debug("login redis index ensured")

    for embedding_index, embedding in enumerate(embeddings):
        _debug(
            "login redis search",
            embedding_index=embedding_index,
            shape=getattr(embedding, "shape", None),
        )
        _debug("login redis keys", count=_speaker_key_count())
        result = redis_store_service.find_similar_users(embedding, top_k=1)
        if _has_matching_user(result, embedding_index=embedding_index):
            _debug("login accepted", embedding_index=embedding_index)
            return jsonify({"message": "logged in"}), 200

    _debug("login rejected", reason="no matching users", embeddings=len(embeddings))
    return jsonify({"error": "unauthorized"}), 401


def _has_matching_user(result, embedding_index: int | None = None) -> bool:
    docs = getattr(result, "docs", [])
    _debug(
        "redis search result",
        embedding_index=embedding_index,
        total=getattr(result, "total", None),
        docs=len(docs),
    )

    for document in docs:
        score = float(document.score)
        matched = score <= face_match_threshold
        _debug(
            "redis match candidate",
            embedding_index=embedding_index,
            login=getattr(document, "login", None),
            score=score,
            threshold=face_match_threshold,
            matched=matched,
        )
        if score <= face_match_threshold:
            return True

    return False


def _debug(message: str, **values) -> None:
    details = " ".join(f"{key}={value}" for key, value in values.items())
    suffix = f" {details}" if details else ""

    print(f"[api-debug] {message}{suffix}", flush=True)


def _uploaded_file_count() -> int:
    return sum(len(request.files.getlist(field_name)) for field_name in request.files)


def _file_names(file_paths: list[str]) -> list[str]:
    return [os.path.basename(file_path) for file_path in file_paths]


def _speaker_key_count() -> int:
    return sum(1 for _ in redis_store_service.redis.scan_iter(match="speaker:*"))


@app.route("/ping")
def ping():
    return jsonify({"ok": True})


@app.post("/clear")
def clear():
    deleted = redis_store_service.clear_profiles()

    return jsonify({"message": "cleared", "deleted": deleted}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
