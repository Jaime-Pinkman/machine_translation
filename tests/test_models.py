import numpy as np
import pytest

from config import VocabularyConfig, TokenizerConfig, COMBINATIONS
from src.models.sentence_classifier.tokenizer import Tokenizer, Vocabulary
from src.models.sentence_classifier.vectorizer import VectorizerFactory, BaseVectorizer
from tests.data_fixtures import (
    expected_outputs,
    expected_tokens,
    expected_freqs,
    test_data,
    vocab_config,
    tokenizer_config,
)
from tests.model_fixtures import (
    tokenizer,
    vocabulary,
    vectorizer
)


def test_tokenizer(
    tokenizer: Tokenizer,
    test_data: list[str],
    expected_tokens: list[list[str]],
) -> None:
    tokenized_data = tokenizer.tokenize_corpus(test_data)
    assert tokenized_data == expected_tokens


def test_vocabulary(
    vocabulary: Vocabulary,
    expected_tokens: list[list[str]],
    expected_freqs: list[float]
) -> None:
    vocabulary.build(expected_tokens)
    assert np.alltrue(
        vocabulary.get_freqs() == expected_freqs
    )


def test_vectorizer(
    vectorizer: BaseVectorizer,
    tokenizer: Tokenizer,
    test_data: list[str],
    expected_outputs: dict[tuple[str, str | None], list[list[float]]],
) -> None:
    tokenized_data = tokenizer.tokenize_corpus(test_data)
    pred_output = vectorizer.vectorize(tokenized_data)
    true_output = expected_outputs[(vectorizer.mode, vectorizer.scale)]
    if not isinstance(pred_output, np.ndarray):
        pred_output = pred_output.todense()
    assert np.allclose(pred_output, true_output, atol=0.01)
