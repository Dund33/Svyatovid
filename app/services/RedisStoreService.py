from __future__ import annotations

import bcrypt
from dataclasses import dataclass
import numpy as np
import redis

from models.UserProfile import UserProfile

from redis.commands.search.field import TextField, VectorField

from redis.commands.search.index_definition import IndexDefinition, IndexType

from redis.exceptions import ResponseError


@dataclass(frozen=True)
class SimilarUserDocument:
    id: str
    login: str
    score: str


@dataclass(frozen=True)
class SimilarUsersResult:
    total: int
    docs: list[SimilarUserDocument]


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

    def get_profile(self, login: str) -> UserProfile | None:

        data = self.redis.hgetall(self._key(login))

        if not data:
            return None

        return UserProfile(
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

    def find_similar_users(self, embedding: np.ndarray, top_k: int = 1):
        result = self.redis.execute_command(
            "FT.SEARCH",
            self.index_name,
            f"*=>[KNN {top_k} @embedding $vec AS score]",
            "PARAMS",
            2,
            "vec",
            self._to_bytes(embedding),
            "RETURN",
            2,
            "login",
            "score",
            "SORTBY",
            "score",
            "DIALECT",
            2,
        )

        return self._parse_search_result(result)

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

    @classmethod
    def _parse_search_result(cls, result) -> SimilarUsersResult:
        if isinstance(result, dict):
            return cls._parse_dict_search_result(result)

        return cls._parse_list_search_result(result)

    @classmethod
    def _parse_dict_search_result(cls, result: dict) -> SimilarUsersResult:
        docs = []

        for document in cls._dict_get(result, "results", []):
            attributes = cls._dict_get(document, "extra_attributes", {})
            docs.append(
                SimilarUserDocument(
                    id=cls._decode(cls._dict_get(document, "id", "")),
                    login=cls._decode(cls._dict_get(attributes, "login", "")),
                    score=cls._decode(cls._dict_get(attributes, "score", "")),
                )
            )

        return SimilarUsersResult(
            total=int(cls._dict_get(result, "total_results", len(docs))),
            docs=docs,
        )

    @classmethod
    def _parse_list_search_result(cls, result: list) -> SimilarUsersResult:
        if not result:
            return SimilarUsersResult(total=0, docs=[])

        docs = []
        for index in range(1, len(result), 2):
            attributes = cls._field_list_to_dict(result[index + 1])
            docs.append(
                SimilarUserDocument(
                    id=cls._decode(result[index]),
                    login=cls._decode(cls._dict_get(attributes, "login", "")),
                    score=cls._decode(cls._dict_get(attributes, "score", "")),
                )
            )

        return SimilarUsersResult(total=int(result[0]), docs=docs)

    @staticmethod
    def _field_list_to_dict(fields: list) -> dict:
        return {
            fields[index]: fields[index + 1]
            for index in range(0, len(fields), 2)
        }

    @staticmethod
    def _dict_get(payload: dict, key: str, default=None):
        return payload.get(key, payload.get(key.encode(), default))

    @staticmethod
    def _decode(value) -> str:
        if isinstance(value, bytes):
            return value.decode()

        return str(value)
