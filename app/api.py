import os

from flask import Flask, jsonify, request

from services.RedisStoreService import RedisStoreService
from services.UploadService import UploadService
from tools import create_user_embedding_from_images


app = Flask(__name__)

upload_service = UploadService(os.getenv("UPLOAD_FOLDER", "/app/uploads"))
redis_store_service = RedisStoreService(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    embedding_dim=int(os.getenv("EMBEDDING_DIM", "512")),
)


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


@app.route("/ping")
def ping():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
