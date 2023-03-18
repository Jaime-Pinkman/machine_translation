import numpy as np
import scipy.sparse


class Vectorizer:
    def __init__(
        self, tokenizer, vocabulary, mode="ltfidf", scale="minmax", use_sparse=False
    ):
        self.tokenizer = tokenizer
        self.vocab = vocabulary
        assert mode in {"ltfidf", "tfidf", "idf", "tf", "bin"}
        assert scale in {"minmax", "std", None}
        self.mode = mode
        self.scale = scale
        self.use_sparse = use_sparse

    def fit(self, texts):
        if self.vocab.word2id is None or self.vocab.word2freq is None:
            tokenized_texts = self.tokenizer.tokenize_corpus(texts)
            self.vocab.build(tokenized_texts)

    def transform(self, texts):
        if self.vocab.word2id is None or self.vocab.word2freq is None:
            raise ValueError("Vectorizer has not been fitted yet.")
        tokenized_texts = self.tokenizer.tokenize_corpus(texts)
        if self.use_sparse:
            result = scipy.sparse.dok_matrix(
                (len(tokenized_texts), len(self.vocab.word2id)), dtype="float32"
            )
        else:
            result = np.zeros((len(tokenized_texts), len(self.vocab.word2id)))

        for text_i, text in enumerate(tokenized_texts):
            for token in text:
                if token in self.vocab.word2id:
                    result[text_i, self.vocab.word2id[token]] += 1

        if self.mode == "bin":
            result = (result > 0).astype("float32")

        elif self.mode == "tf":
            if self.use_sparse:
                result = result.tocsr()
                result = result.multiply(1 / result.sum(1))
            else:
                result = result * (1 / result.sum(1))[:, np.newaxis]

        elif self.mode == "idf":
            if self.use_sparse:
                result = (
                    (result > 0).astype("float32").multiply(1 / self.vocab.word2freq)
                )
            else:
                result = (result > 0).astype("float32") * (1 / self.vocab.word2freq)

        elif self.mode == "tfidf":
            if self.use_sparse:
                result = result.tocsr()
                result = result.multiply(1 / result.sum(1))
                result = result.multiply(1 / self.vocab.word2freq)
            else:
                result = result * (1 / result.sum(1))[:, np.newaxis]
                result = result * (1 / self.vocab.word2freq)

        elif self.mode == "ltfidf":
            if self.use_sparse:
                result = result.tocsr()
                result = result.multiply(1 / result.sum(1)).log1p()
                result = result.multiply(1 / self.vocab.word2freq)
            else:
                result = np.log(result * (1 / result.sum(1))[:, np.newaxis] + 1)
                result = result * (1 / self.vocab.word2freq)

        if self.scale == "minmax":
            if self.use_sparse:
                result = result.tocsc()
            result -= result.min()
            result /= result.max() + 1e-6

        elif self.scale == "std":
            if self.use_sparse:
                result = result.tocsc()
            result -= result.mean(0)
            result /= result.std(0, ddof=1)

        if self.use_sparse:
            result = result.tocsr()

        return result
