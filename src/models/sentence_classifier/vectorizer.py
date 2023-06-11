from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse

from config import EmbeddingMode, ScaleType
from src.models.sentence_classifier.tokenizer import Vocabulary, Tokenizer


class BaseVectorizer(ABC):
    def __init__(
        self,
        tokenizer: Tokenizer,
        vocabulary: Vocabulary,
        mode: str = "ltfidf",
        scale: str = "minmax",
    ):
        self.tokenizer = tokenizer
        self.vocab = vocabulary
        assert mode in {"ltfidf", "tfidf", "idf", "tf", "bin"}
        assert scale in {"minmax", "std", None}
        self.mode = mode
        self.scale = scale

    def fit(self, texts: list[str]) -> None:
        if self.vocab.word2id is None or self.vocab.word2freq is None:
            tokenized_texts = self.tokenizer.tokenize_corpus(texts)
            self.vocab.build(tokenized_texts)

    @abstractmethod
    def _initialize_result_array(
        self, n_rows: int, n_cols: int
    ) -> np.ndarray | sparse._dok.dok_matrix:
        pass

    @abstractmethod
    def _apply_mode(
        self, result: np.ndarray | sparse._dok.dok_matrix
    ) -> np.ndarray | sparse._dok.dok_matrix:
        pass

    @abstractmethod
    def _apply_scaling(
        self, result: np.ndarray | sparse._dok.dok_matrix
    ) -> np.ndarray | sparse._dok.dok_matrix:
        pass

    def transform(self, texts: list[str]) -> np.ndarray | sparse._dok.dok_matrix:
        if self.vocab.word2id is None or self.vocab.word2freq is None:
            raise ValueError("Vectorizer has not been fit yet.")
        tokenized_texts = self.tokenizer.tokenize_corpus(texts)
        result = self._initialize_result_array(
            len(tokenized_texts), len(self.vocab.word2id)
        )

        for text_i, text in enumerate(tokenized_texts):
            for token in text:
                if token in self.vocab.word2id:
                    result[text_i, self.vocab.word2id[token]] += 1

        result = self._apply_mode(result)

        result = self._apply_scaling(result)

        return result


class SparseVectorizer(BaseVectorizer):
    def _initialize_result_array(
        self, n_rows: int, n_cols: int
    ) -> sparse._dok.dok_matrix:
        return sparse.dok_matrix((n_rows, n_cols), dtype="float32")

    def _apply_mode(self, result: sparse._dok.dok_matrix) -> sparse._dok.dok_matrix:
        match self.mode:
            case EmbeddingMode.BIN.value:
                result = (result > 0).astype("float32")

            case EmbeddingMode.TF.value:
                result = result.tocsr()
                result = result.multiply(1 / result.sum(1))

            case EmbeddingMode.IDF.value:
                result = (
                    (result > 0).astype("float32").multiply(1 / self.vocab.get_freqs())
                )

            case EmbeddingMode.TFIDF.value:
                result = result.tocsr()
                result = result.multiply(1 / result.sum(1))
                result = result.multiply(1 / self.vocab.get_freqs())

            case EmbeddingMode.LTFIDF.value:
                result = result.tocsr()
                result = result.multiply(1 / result.sum(1)).log1p()
                result = result.multiply(1 / self.vocab.get_freqs())

        return result

    def _apply_scaling(self, result: sparse._dok.dok_matrix) -> sparse._dok.dok_matrix:
        match self.scale:
            case ScaleType.MINMAX.value:
                result = result.tocsc()
                result -= result.min()
                result /= result.max() + 1e-6

            case ScaleType.STD.value:
                result = result.tocsc()
                result -= result.mean(0)
                result /= result.std(0, ddof=1)
                result = sparse.dok_matrix(result, dtype="float32")

        return result.tocsr()


class DenseVectorizer(BaseVectorizer):
    def _initialize_result_array(self, n_rows: int, n_cols: int) -> np.ndarray:
        return np.zeros((n_rows, n_cols))

    def _apply_mode(self, result: np.ndarray) -> np.ndarray:
        match self.mode:
            case EmbeddingMode.BIN.value:
                result = (result > 0).astype("float32")

            case EmbeddingMode.TF.value:
                result = result * (1 / result.sum(1))[:, np.newaxis]

            case EmbeddingMode.IDF.value:
                result = (result > 0).astype("float32") * (1 / self.vocab.get_freqs())

            case EmbeddingMode.TFIDF.value:
                result = result * (1 / result.sum(1))[:, np.newaxis]
                result = result * (1 / self.vocab.get_freqs())

            case EmbeddingMode.LTFIDF.value:
                result = np.log(result * (1 / result.sum(1))[:, np.newaxis] + 1)
                result = result * (1 / self.vocab.get_freqs())

        return result

    def _apply_scaling(self, result: np.ndarray) -> np.ndarray:
        match self.scale:
            case ScaleType.MINMAX.value:
                result -= result.min()
                result /= result.max() + 1e-6

            case ScaleType.STD.value:
                result -= result.mean(0)
                result /= result.std(0, ddof=1)

        return result


class VectorizerFactory:
    def __init__(
        self,
        vocabulary: Vocabulary,
        mode: str = "ltfidf",
        scale: str = "minmax",
        use_sparse: bool = False,
    ):
        self.vocabulary = vocabulary
        self.mode = mode
        self.scale = scale
        self.use_sparse = use_sparse

    def get_vectorizer(self, tokenizer: Tokenizer) -> BaseVectorizer:
        if self.use_sparse:
            return SparseVectorizer(
                tokenizer=tokenizer,
                vocabulary=self.vocabulary,
                mode=self.mode,
                scale=self.scale,
            )
        else:
            return DenseVectorizer(
                tokenizer=tokenizer,
                vocabulary=self.vocabulary,
                mode=self.mode,
                scale=self.scale,
            )
