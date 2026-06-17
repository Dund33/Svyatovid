import os

from flask import Flask, jsonify, request

from services.RedisStoreService import RedisStoreService
from services.UploadService import UploadService
from ai_tools import face_embeddings_from_images
from tools import create_user_embedding_from_images


app = Flask(__name__)

upload_service = UploadService(os.getenv("UPLOAD_FOLDER", "/app/uploads"))
redis_store_service = RedisStoreService(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    embedding_dim=int(os.getenv("EMBEDDING_DIM", "512")),
)
face_match_threshold = float(os.getenv("FACE_MATCH_THRESHOLD", "0.3"))


@app.post("/register")
def register():
    username = request.form.get("username")
    password = request.form.get("password")

    if not username or not password:
        return jsonify({"error": "username and password required"}), 400

    if not request.files:
        return jsonify({"error": "at least one image file required"}), 400

    if redis_store_service.get_profile(username):
        return jsonify({"error": "user already exists"}), 409

    file_paths = upload_service.save_files(request.files)

    if not file_paths:
        return jsonify({"error": "at least one image file required"}), 400

    try:
        embedding = create_user_embedding_from_images(file_paths)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    redis_store_service.ensure_index()
    redis_store_service.create_profile(
        login=username,
        password=password,
        embedding=embedding,
    )

    return jsonify({"message": "registered", "username": username}), 201


@app.post("/login")
def login():
    if not request.files:
        return jsonify({"error": "image file required"}), 400

    file_paths = upload_service.save_files(request.files)

    if not file_paths:
        return jsonify({"error": "image file required"}), 400

    try:
        embeddings = face_embeddings_from_images(file_paths)
    except ValueError:
        return jsonify({"error": "unauthorized"}), 401

    if not embeddings:
        return jsonify({"error": "unauthorized"}), 401

    redis_store_service.ensure_index()

    for embedding in embeddings:
        result = redis_store_service.find_similar_users(embedding, top_k=1)
        if _has_matching_user(result):
            return jsonify({"message": "logged in"}), 200

    return jsonify({"error": "unauthorized"}), 401


def _has_matching_user(result) -> bool:
    for document in getattr(result, "docs", []):
        score = float(document.score)
        if score <= face_match_threshold:
            return True

    return False


@app.route("/ping")
def ping():
    return jsonify({"ok": True})


@app.post("/clear")
def clear():
    deleted = redis_store_service.clear_profiles()

    return jsonify({"message": "cleared", "deleted": deleted}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
