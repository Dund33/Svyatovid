from __future__ import annotations

import bcrypt
import numpy as np
import redis

from src.models.SpeakerProfile import SpeakerProfile

from redis.commands.search.field import TextField, VectorField

from redis.commands.search.index_definition import IndexDefinition, IndexType

from redis.commands.search.query import Query
from redis.exceptions import ResponseError


class RedisStoreService:
    def __init__(
        self, redis_url: str, embedding_dim: int = 256, index_name: str = "speaker_idx"
    ):

        self.redis = redis.Redis.from_url(redis_url, decode_responses=False)

        self.embedding_dim = embedding_dim
        self.index_name = index_name

    def _key(self, login: str) -> str:
        return f"speaker:{login}"

    @staticmethod
    def _to_bytes(embedding: np.ndarray) -> bytes:

        return np.asarray(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def _from_bytes(payload: bytes) -> np.ndarray:

        return np.frombuffer(payload, dtype=np.float32)

    def ensure_index(self):

        schema = (
            TextField("login"),
            TextField("password_hash"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.embedding_dim,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )

        try:
            self.redis.ft(self.index_name).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=["speaker:"], index_type=IndexType.HASH
                ),
            )

        except ResponseError as e:
            if "Index already exists" not in str(e):
                raise

    def get_profile(self, login: str) -> SpeakerProfile | None:

        data = self.redis.hgetall(self._key(login))

        if not data:
            return None

        return SpeakerProfile(
            login=data[b"login"].decode(),
            password_hash=data[b"password_hash"].decode(),
            embedding=self._from_bytes(data[b"embedding"]),
        )

    def create_profile(self, login: str, password: str, embedding: np.ndarray):

        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        self.redis.hset(
            self._key(login),
            mapping={
                "login": login,
                "password_hash": password_hash,
                "embedding": self._to_bytes(embedding),
            },
        )

    def find_similar_speakers(self, embedding: np.ndarray, top_k: int = 1):

        query = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
            .return_fields("login", "score")
            .sort_by("score")
            .dialect(2)
        )

        return self.redis.ft(self.index_name).search(
            query, query_params={"vec": self._to_bytes(embedding)}
        )

    def clear_profiles(self):
        """
        Usuwa wszystkie rekordy speaker:* z Redis.
        Nie usuwa indeksu RediSearch.

        Returns:
            int: liczba usuniętych rekordów.
        """
        pattern = "speaker:*"
        cursor = 0
        deleted = 0

        while True:
            cursor, keys = self.redis.scan(cursor=cursor, match=pattern, count=100)

            if keys:
                deleted += self.redis.delete(*keys)

            if cursor == 0:
                break

        return deleted
