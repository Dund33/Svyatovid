from dataclasses import dataclass

import numpy as np


@dataclass
class UserProfile:
    login: str
    password_hash: str
    embedding: np.ndarray
