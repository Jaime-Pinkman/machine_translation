import collections
import re

import numpy as np


class Tokenizer:

    TOKEN_RE = re.compile(r"[\w\d]+")

    def __init__(self, min_token_size=4):
        self.min_token_size = min_token_size

    def __call__(self, text):
        text = text.lower()
        all_tokens = self.TOKEN_RE.findall(text)
        return [token for token in all_tokens if len(token) >= self.min_token_size]

    def tokenize_corpus(self, texts):
        return [self(text) for text in texts]


class Vocabulary:
    def __init__(
        self, max_size=1_000_000, max_doc_freq=0.8, min_count=5, pad_word=None
    ):
        self.max_size = max_size
        self.max_doc_freq = max_doc_freq
        self.min_count = min_count
        self.pad_word = pad_word
        self.word2id = None
        self.word2freq = None

    def build(self, tokenized_texts):
        word_counts = collections.defaultdict(int)
        _doc_n = 0
        for _doc_n, txt in enumerate(tokenized_texts, start=1):
            unique_text_tokens = set(txt)
            for token in unique_text_tokens:
                word_counts[token] += 1

        word_counts = {
            word: cnt
            for word, cnt in word_counts.items()
            if cnt >= self.min_count and cnt / _doc_n < self.max_doc_freq
        }

        sorted_word_counts = sorted(
            word_counts.items(), reverse=True, key=lambda pair: pair[1]
        )

        if self.pad_word is not None:
            sorted_word_counts = [(self.pad_word, 0)] + sorted_word_counts

        if len(word_counts) > self.max_size:
            sorted_word_counts = sorted_word_counts[: self.max_size]

        self.word2id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}

        self.word2freq = np.array(
            [cnt / _doc_n for _, cnt in sorted_word_counts], dtype="float32"
        )

    def texts_to_token_ids(self, tokenized_texts):
        return [
            [self.word2id[token] for token in text if token in self.word2id]
            for text in tokenized_texts
        ]

    def get_sorted_token_freq(self):
        word_df = [
            (key, freq) for key, freq in zip(self.word2id.keys(), self.word2freq)
        ]
        return sorted(word_df, key=lambda x: (x[1], x[0]))
