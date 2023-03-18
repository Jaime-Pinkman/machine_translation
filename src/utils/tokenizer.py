import collections
import re

import numpy as np

TOKEN_RE = re.compile(r"[\w\d]+")


def tokenize_text_simple_regex(txt, min_token_size=4):
    txt = txt.lower()
    all_tokens = TOKEN_RE.findall(txt)
    return [token for token in all_tokens if len(token) >= min_token_size]


def character_tokenize(txt):
    return list(txt)


def tokenize_corpus(texts, tokenizer=tokenize_text_simple_regex, **tokenizer_kwargs):
    return [tokenizer(text, **tokenizer_kwargs) for text in texts]


def add_fake_token(word2id, token="<PAD>"):
    word2id_new = {token: i + 1 for token, i in word2id.items()}
    word2id_new[token] = 0
    return word2id_new


def text_to_token_ids(tokenized_texts, word2id):
    return [
        [word2id[token] for token in text if token in word2id]
        for text in tokenized_texts
    ]


def build_vocabulary(
    tokenized_texts, max_size=1_000_000, max_doc_freq=0.8, min_count=5, pad_word=None
):
    word_counts = collections.defaultdict(int)

    for _doc_n, txt in enumerate(tokenized_texts):
        unique_text_tokens = set(txt)
        for token in unique_text_tokens:
            word_counts[token] += 1

    word_counts = {
        word: cnt
        for word, cnt in word_counts.items()
        if cnt >= min_count and cnt / _doc_n < max_doc_freq
    }

    sorted_word_counts = sorted(
        word_counts.items(), reverse=True, key=lambda pair: pair[1]
    )

    if pad_word is not None:
        sorted_word_counts = [(pad_word, 0)] + sorted_word_counts

    if len(word_counts) > max_size:
        sorted_word_counts = sorted_word_counts[:max_size]

    word2id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}

    word2freq = np.array(
        [cnt / _doc_n for _, cnt in sorted_word_counts], dtype="float32"
    )

    return word2id, word2freq


def get_sorted_token_freq(vocabulary, word_doc_freq):
    word_df = [(key, freq) for key, freq in zip(vocabulary.keys(), word_doc_freq)]
    return sorted(word_df, key=lambda x: (x[1], x[0]))
