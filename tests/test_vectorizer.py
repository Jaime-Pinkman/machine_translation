import numpy as np
import pytest

from config import VocabularyConfig, TokenizerConfig
from src.models.sentence_classifier.tokenizer import Tokenizer, Vocabulary
from src.models.sentence_classifier.vectorizer import VectorizerFactory, BaseVectorizer
from tests.test_data import (
    COMBINATIONS,
    expected_outputs,
    expected_tokens,
    test_data,
    vocab_config,
    tokenizer_config,
)


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
    tokenizer: Tokenizer,
    vocabulary: Vocabulary,
) -> BaseVectorizer:
    mode, scale, use_sparse = request.param
    vocabulary.build(expected_tokens)
    vectorizer_factory = VectorizerFactory(
        vocabulary, mode=mode, scale=scale, use_sparse=use_sparse
    )
    vectorizer = vectorizer_factory.get_vectorizer(tokenizer)
    return vectorizer


def test_vectorizer(
    vectorizer: BaseVectorizer,
    test_data: list[str],
    expected_outputs: dict[tuple[str, str | None], list[list[float]]],
) -> None:
    pred_output = vectorizer.transform(test_data)
    true_output = expected_outputs[(vectorizer.mode, vectorizer.scale)]
    if not isinstance(pred_output, np.ndarray):
        pred_output = pred_output.todense()
    assert np.allclose(pred_output, true_output, atol=0.01)
