import pytest

from text_classifier.data.text_processing import Tokenizer, Vocabulary, BaseVectorizer
from text_classifier.data.text_processing.embedding_enums import COMBINATIONS
from tests.data_fixtures import (
    expected_outputs,
    expected_tokens,
    expected_freqs,
    test_data,
)
from tests.model_fixtures import tokenizer, vocabulary, vectorizer


@pytest.fixture
def test_data_fixture() -> list[str]:
    return test_data()


@pytest.fixture
def expected_tokens_fixture() -> list[list[str]]:
    return expected_tokens()


@pytest.fixture
def expected_freqs_fixture() -> list[float]:
    return expected_freqs()


@pytest.fixture
def expected_outputs_fixture() -> dict[tuple[str, str | None], list[list[float]]]:
    return expected_outputs()


@pytest.fixture
def tokenizer_fixture() -> Tokenizer:
    return tokenizer()


@pytest.fixture
def vocabulary_fixture() -> Vocabulary:
    return vocabulary()


@pytest.fixture(params=COMBINATIONS)
def vectorizer_fixture(
    request: pytest.FixtureRequest,
    expected_tokens_fixture: list[list[str]],
    vocabulary_fixture: Vocabulary,
) -> BaseVectorizer:
    return vectorizer(request, expected_tokens_fixture, vocabulary_fixture)
