import pytest

from config import VocabularyConfig, TokenizerConfig, COMBINATIONS
from src.models.sentence_classifier.tokenizer import Tokenizer, Vocabulary
from src.models.sentence_classifier.vectorizer import VectorizerFactory, BaseVectorizer


@pytest.fixture
def tokenizer(tokenizer_config: TokenizerConfig) -> Tokenizer:
    tknzr = Tokenizer(min_token_size=tokenizer_config.min_token_size)
    return tknzr


@pytest.fixture
def vocabulary(vocab_config: VocabularyConfig) -> Vocabulary:
    vocab = Vocabulary(
        max_size=vocab_config.max_size,
        max_doc_freq=vocab_config.max_doc_freq,
        min_count=vocab_config.min_count,
        pad_word=vocab_config.pad_word,
    )
    return vocab


@pytest.fixture(params=COMBINATIONS)
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
