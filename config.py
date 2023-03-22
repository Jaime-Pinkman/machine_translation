from enum import Enum

vocab_config = {
    "max_size": 10000,
    "max_doc_freq": 1,
    "min_count": 1,
    "pad_word": None,
}
tokenizer_config = {
    "min_token_size": 0,
}


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
