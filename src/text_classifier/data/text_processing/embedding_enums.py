from enum import Enum
from itertools import product


class EmbeddingMode(Enum):
    BIN = "bin"
    TF = "tf"
    IDF = "idf"
    TFIDF = "tfidf"
    LTFIDF = "ltfidf"


class ScaleType(Enum):
    MINMAX = "minmax"
    STD = "std"
    NONE = None


COMBINATIONS = list(
    product(
        [e.value for e in EmbeddingMode], [s.value for s in ScaleType], [False, True]
    )
)
