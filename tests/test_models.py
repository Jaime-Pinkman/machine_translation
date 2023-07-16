import numpy as np

from text_classifier.data.text_processing import Tokenizer, Vocabulary, BaseVectorizer


def test_tokenizer(
    tokenizer_fixture: Tokenizer,
    test_data_fixture: list[str],
    expected_tokens_fixture: list[list[str]],
) -> None:
    tokenized_data = tokenizer_fixture.tokenize_corpus(test_data_fixture)
    assert tokenized_data == expected_tokens_fixture


def test_vocabulary(
    vocabulary_fixture: Vocabulary,
    expected_tokens_fixture: list[list[str]],
    expected_freqs_fixture: list[float],
) -> None:
    vocabulary_fixture.build(expected_tokens_fixture)
    assert np.alltrue(vocabulary_fixture.get_freqs() == expected_freqs_fixture)


def test_vectorizer(
    vectorizer_fixture: BaseVectorizer,
    tokenizer_fixture: Tokenizer,
    test_data_fixture: list[str],
    expected_outputs_fixture: dict[tuple[str, str | None], list[list[float]]],
) -> None:
    tokenized_data = tokenizer_fixture.tokenize_corpus(test_data_fixture)
    pred_output = vectorizer_fixture.vectorize(tokenized_data)
    true_output = expected_outputs_fixture[
        (vectorizer_fixture.mode, vectorizer_fixture.scale)
    ]
    if not isinstance(pred_output, np.ndarray):
        pred_output = pred_output.todense()
    assert np.allclose(pred_output, true_output, atol=0.01)
