import pytest

from config import VocabularyConfig, TokenizerConfig
from src.models import Tokenizer, Vocabulary, VectorizerFactory, BaseVectorizer


def tokenizer() -> Tokenizer:
    tknzr = Tokenizer(min_token_size=TokenizerConfig.min_token_size)
    return tknzr


def vocabulary() -> Vocabulary:
    vocab = Vocabulary(
        max_size=VocabularyConfig.max_size,
        max_doc_freq=VocabularyConfig.max_doc_freq,
        min_count=VocabularyConfig.min_count,
        pad_word=VocabularyConfig.pad_word,
    )
    return vocab


def vectorizer(
    request: pytest.FixtureRequest,
    expected_tokens: list[list[str]],
    vocabulary: Vocabulary,
) -> BaseVectorizer:
    mode, scale, use_sparse = request.param
    vocabulary.build(expected_tokens)
    vectorizer_factory = VectorizerFactory(
        vocabulary, mode=mode, scale=scale, use_sparse=use_sparse
    )
    vectorizer = vectorizer_factory.get_vectorizer()
    return vectorizer
