import numpy as np
from scipy import sparse


class VectorizerFactory:
    def __init__(self, vocabulary, mode="ltfidf", scale="minmax", use_sparse=False):
        self.vocabulary = vocabulary
        self.mode = mode
        self.scale = scale
        self.use_sparse = use_sparse

    def get_vectorizer(self, tokenizer):
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


class BaseVectorizer:
    def __init__(self, tokenizer, vocabulary, mode="ltfidf", scale="minmax"):
        self.tokenizer = tokenizer
        self.vocab = vocabulary
        assert mode in {"ltfidf", "tfidf", "idf", "tf", "bin"}
        assert scale in {"minmax", "std", None}
        self.mode = mode
        self.scale = scale

    def fit(self, texts):
        if self.vocab.word2id is None or self.vocab.word2freq is None:
            tokenized_texts = self.tokenizer.tokenize_corpus(texts)
            self.vocab.build(tokenized_texts)

    def transform(self, texts):
        if self.vocab.word2id is None or self.vocab.word2freq is None:
            raise ValueError("Vectorizer has not been fitted yet.")
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

    def _initialize_result_array(self, n_rows, n_cols):
        raise NotImplementedError

    def _apply_mode(self, result):
        raise NotImplementedError

    def _apply_scaling(self, result):
        raise NotImplementedError


class SparseVectorizer(BaseVectorizer):
    def _initialize_result_array(self, n_rows, n_cols):
        return sparse.dok_matrix((n_rows, n_cols), dtype="float32")

    def _apply_mode(self, result):
        if self.mode == "bin":
            result = (result > 0).astype("float32")

        elif self.mode == "tf":
            result = result.tocsr()
            result = result.multiply(1 / result.sum(1))

        elif self.mode == "idf":
            result = (result > 0).astype("float32").multiply(1 / self.vocab.word2freq)

        elif self.mode == "tfidf":
            result = result.tocsr()
            result = result.multiply(1 / result.sum(1))
            result = result.multiply(1 / self.vocab.word2freq)

        elif self.mode == "ltfidf":
            result = result.tocsr()
            result = result.multiply(1 / result.sum(1)).log1p()
            result = result.multiply(1 / self.vocab.word2freq)

        return result

    def _apply_scaling(self, result):
        if self.scale == "minmax":
            result = result.tocsc()
            result -= result.min()
            result /= result.max() + 1e-6

        elif self.scale == "std":
            result = result.tocsc()
            result -= result.mean(0)
            result /= result.std(0, ddof=1)
            result = sparse.dok_matrix(result, dtype="float32")

        return result.tocsr()


class DenseVectorizer(BaseVectorizer):
    def _initialize_result_array(self, n_rows, n_cols):
        return np.zeros((n_rows, n_cols))

    def _apply_mode(self, result):
        if self.mode == "bin":
            result = (result > 0).astype("float32")

        elif self.mode == "tf":
            result = result * (1 / result.sum(1))[:, np.newaxis]

        elif self.mode == "idf":
            result = (result > 0).astype("float32") * (1 / self.vocab.word2freq)

        elif self.mode == "tfidf":
            result = result * (1 / result.sum(1))[:, np.newaxis]
            result = result * (1 / self.vocab.word2freq)

        elif self.mode == "ltfidf":
            result = np.log(result * (1 / result.sum(1))[:, np.newaxis] + 1)
            result = result * (1 / self.vocab.word2freq)

        return result

    def _apply_scaling(self, result):
        if self.scale == "minmax":
            result -= result.min()
            result /= result.max() + 1e-6

        elif self.scale == "std":
            result -= result.mean(0)
            result /= result.std(0, ddof=1)

        return result
