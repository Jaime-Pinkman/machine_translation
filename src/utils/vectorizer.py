import numpy as np
import scipy.sparse


def vectorize_texts_sparse(
    tokenized_texts, word2id, word2freq, mode="tfidf", scale=True
):
    assert mode in {"tfidf", "idf", "tf", "bin"}

    result = scipy.sparse.dok_matrix(
        (len(tokenized_texts), len(word2id)), dtype="float32"
    )
    for text_i, text in enumerate(tokenized_texts):
        for token in text:
            if token in word2id:
                result[text_i, word2id[token]] += 1

    if mode == "bin":
        result = (result > 0).astype("float32")

    elif mode == "tf":
        result = result.tocsr()
        result = result.multiply(1 / result.sum(1))

    elif mode == "idf":
        result = (result > 0).astype("float32").multiply(1 / word2freq)

    elif mode == "tfidf":
        result = result.tocsr()
        result = result.multiply(1 / result.sum(1))
        result = result.multiply(1 / word2freq)

    if scale:
        result = result.tocsc()
        result -= result.min()
        result /= result.max() + 1e-6

    return result.tocsr()


def vectorize_texts(tokenized_texts, word2id, word2freq, mode="tfidf", scale="minmax"):
    assert mode in {"ltfidf", "tfidf", "idf", "tf", "bin"}
    assert scale in {"minmax", "std", None}

    result = np.zeros((len(tokenized_texts), len(word2id)))
    for text_i, text in enumerate(tokenized_texts):
        for token in text:
            if token in word2id:
                result[text_i, word2id[token]] += 1

    if mode == "bin":
        result = (result > 0).astype("float32")

    elif mode == "tf":
        result = result.multiply(1 / result.sum(1))

    elif mode == "idf":
        result = (result > 0).astype("float32").multiply(1 / word2freq)

    elif mode == "tfidf":
        result = result.multiply(1 / result.sum(1))
        result = result.multiply(1 / word2freq)

    elif mode == "ltfidf":
        result = np.log(result * (1 / result.sum(1))[:, np.newaxis] + 1)
        result = result * (1 / word2freq)

    if scale == "minmax":
        result -= result.min()
        result /= result.max() + 1e-6

    if scale == "std":
        result -= result.mean(0)
        result /= result.std(0, ddof=1)

    return result
