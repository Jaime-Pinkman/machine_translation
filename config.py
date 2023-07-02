from dataclasses import dataclass
from enum import Enum
from itertools import product


@dataclass
class VocabularyConfig:
    max_size: int = 10_000
    max_doc_freq: float = 1.0
    min_count: int = 1
    pad_word: str | None = None
    word2id: dict[str, int] | None = None
    word2freq: dict[str, float] | None = None


@dataclass
class TokenizerConfig:
    min_token_size: int = 0


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
