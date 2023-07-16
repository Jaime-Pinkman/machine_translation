import collections
import re

import numpy as np

from text_classifier.utils import check_is_fitted


class Tokenizer:

    TOKEN_RE = re.compile(r"[\w\d]+")

    def __init__(self, min_token_size: int = 4):
        self.min_token_size = min_token_size

    def __call__(self, text: str) -> list[str]:
        text = text.lower()
        all_tokens = self.TOKEN_RE.findall(text)
        return [token for token in all_tokens if len(token) >= self.min_token_size]

    def tokenize_corpus(self, texts: list[str]) -> list[list[str]]:
        return [self(text) for text in texts]


class Vocabulary:
    def __init__(
        self,
        max_size: int = 1_000_000,
        max_doc_freq: float = 0.8,
        min_count: int = 5,
        pad_word: str | None = None,
        word2id: dict[str, int] | None = None,
        id2word: dict[int, str] | None = None,
        word2freq: dict[str, float] | None = None,
    ):
        self.max_size = max_size
        self.max_doc_freq = max_doc_freq
        self.min_count = min_count
        self.pad_word = pad_word
        self.word2id = word2id
        self.id2word = id2word
        self.word2freq = word2freq

    def build(self, tokenized_texts: list[list[str]]) -> None:
        word_counts = collections.defaultdict(int)  # type: dict[str, int]
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
            word_counts.items(), reverse=True, key=lambda pair: (pair[1], pair[0])
        )

        if self.pad_word is not None:
            sorted_word_counts = [(self.pad_word, 0)] + sorted_word_counts

        if len(word_counts) > self.max_size:
            sorted_word_counts = sorted_word_counts[: self.max_size]

        self.word2id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}

        self.id2word = {i: word for word, i in self.word2id.items()}

        self.word2freq = {word: cnt / _doc_n for word, cnt in sorted_word_counts}

    @check_is_fitted(["word2freq"])
    def get_freqs(self) -> np.ndarray:
        return np.array(list(self.word2freq.values()), dtype="float32")  # type: ignore

    @check_is_fitted(["word2id"])
    def tokenized_corpus_to_token_ids(
        self, tokenized_texts: list[list[str]]
    ) -> list[list[int]]:
        return [
            [
                self.word2id[token]  # type: ignore
                for token in text
                if token in self.word2id  # type: ignore
            ]
            for text in tokenized_texts
        ]

    @check_is_fitted(["word2id"])
    def __len__(self) -> int:
        return len(self.word2id)  # type: ignore

    @check_is_fitted(["word2id"])
    def __contains__(self, item: str) -> bool:
        return item in self.word2id  # type: ignore

    @check_is_fitted(["id2word", "word2id"])
    def __getitem__(self, key: int | str) -> str | int:
        if isinstance(key, int):
            return self.id2word[key]  # type: ignore
        return self.word2id[key]  # type: ignore
