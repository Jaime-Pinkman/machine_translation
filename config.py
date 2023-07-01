from dataclasses import dataclass
from enum import Enum
from itertools import product


@dataclass
class VocabularyConfig:
    max_size: int
    max_doc_freq: float
    min_count: int
    pad_word: str | None = None
    word2id: dict[str, int] | None = None
    word2freq: dict[str, float] | None = None


@dataclass
class TokenizerConfig:
    min_token_size: int = 4


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


embedding_modes = [e.value for e in EmbeddingMode]
scale_types = [s.value for s in ScaleType]

COMBINATIONS = list(product(embedding_modes, scale_types, [False, True]))
