import numpy as np
import pytest
from config import tokenizer_config, vocab_config
from src.models.sentence_classifier.tokenizer import Tokenizer, Vocabulary
from src.models.sentence_classifier.vectorizer import VectorizerFactory

from tests.test_data import COMBINATIONS, expected_outputs, expected_tokens, test_data


@pytest.fixture
def tokenizer():
    tknzr = Tokenizer(**tokenizer_config)
    return tknzr


@pytest.fixture
def vocabulary():
    vocab = Vocabulary(**vocab_config)
    return vocab


@pytest.fixture(params=COMBINATIONS)
def factory(request, expected_tokens, tokenizer, vocabulary):
    mode, scale, use_sparse = request.param
    vocabulary.build(expected_tokens)
    vectorizer_factory = VectorizerFactory(
        vocabulary, mode=mode, scale=scale, use_sparse=use_sparse
    )
    vectorizer = vectorizer_factory.get_vectorizer(tokenizer)
    return vectorizer


def test_vectorizer_factory(factory, test_data, expected_outputs):
    pred_output = factory.transform(test_data)
    true_output = expected_outputs[(factory.mode, factory.scale)]
    if not isinstance(pred_output, np.ndarray):
        pred_output = pred_output.todense()
    assert np.allclose(pred_output, true_output, atol=0.01)
