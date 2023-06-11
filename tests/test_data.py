import pytest


@pytest.fixture
def test_data() -> list[str]:
    return [
        "Казнить нельзя, помиловать. Нельзя наказывать.",
        "Казнить, нельзя помиловать. Нельзя освободить.",
        "Нельзя не помиловать.",
        "Обязательно освободить.",
    ]


@pytest.fixture
def expected_tokens() -> list[list[str]]:
    return [
        ["казнить", "нельзя", "помиловать", "нельзя", "наказывать"],
        ["казнить", "нельзя", "помиловать", "нельзя", "освободить"],
        ["нельзя", "не", "помиловать"],
        ["обязательно", "освободить"],
    ]


COMBINATIONS = [
    ("bin", None, False),
    ("bin", None, True),
    ("bin", "minmax", False),
    ("bin", "minmax", True),
    ("bin", "std", False),
    ("bin", "std", True),
    ("idf", None, False),
    ("idf", None, True),
    ("idf", "minmax", False),
    ("idf", "minmax", True),
    ("idf", "std", False),
    ("idf", "std", True),
    ("ltfidf", None, False),
    ("ltfidf", None, True),
    ("ltfidf", "minmax", False),
    ("ltfidf", "minmax", True),
    ("ltfidf", "std", False),
    ("ltfidf", "std", True),
    ("tf", None, False),
    ("tf", None, True),
    ("tf", "minmax", False),
    ("tf", "minmax", True),
    ("tf", "std", False),
    ("tf", "std", True),
    ("tfidf", None, False),
    ("tfidf", None, True),
    ("tfidf", "minmax", False),
    ("tfidf", "minmax", True),
    ("tfidf", "std", False),
    ("tfidf", "std", True),
]


@pytest.fixture
def expected_outputs() -> dict[tuple[str, str | None], list[list[float]]]:
    return {
        ("bin", None): [
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ],
        ("bin", "minmax"): [
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ],
        ("bin", "std"): [
            [0.5, 0.5, -0.87, 0.87, -0.5, -0.5, 1.5],
            [0.5, 0.5, 0.87, 0.87, -0.5, -0.5, -0.5],
            [0.5, 0.5, -0.87, -0.87, -0.5, 1.5, -0.5],
            [-1.5, -1.5, 0.87, -0.87, 1.5, -0.5, -0.5],
        ],
        ("idf", None): [
            [1.33, 1.33, 0.0, 2.0, 0.0, 0.0, 4.0],
            [1.33, 1.33, 2.0, 2.0, 0.0, 0.0, 0.0],
            [1.33, 1.33, 0.0, 0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0],
        ],
        ("idf", "minmax"): [
            [0.33, 0.33, 0.0, 0.5, 0.0, 0.0, 1.0],
            [0.33, 0.33, 0.5, 0.5, 0.0, 0.0, 0.0],
            [0.33, 0.33, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
        ],
        ("idf", "std"): [
            [0.5, 0.5, -0.87, 0.87, -0.5, -0.5, 1.5],
            [0.5, 0.5, 0.87, 0.87, -0.5, -0.5, -0.5],
            [0.5, 0.5, -0.87, -0.87, -0.5, 1.5, -0.5],
            [-1.5, -1.5, 0.87, -0.87, 1.5, -0.5, -0.5],
        ],
        ("ltfidf", None): [
            [0.24, 0.45, 0.0, 0.36, 0.0, 0.0, 0.73],
            [0.24, 0.45, 0.36, 0.36, 0.0, 0.0, 0.0],
            [0.38, 0.38, 0.0, 0.0, 0.0, 1.15, 0.0],
            [0.0, 0.0, 0.81, 0.0, 1.62, 0.0, 0.0],
        ],
        ("ltfidf", "minmax"): [
            [0.15, 0.28, 0.0, 0.22, 0.0, 0.0, 0.45],
            [0.15, 0.28, 0.22, 0.22, 0.0, 0.0, 0.0],
            [0.24, 0.24, 0.0, 0.0, 0.0, 0.71, 0.0],
            [0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
        ],
        ("ltfidf", "std"): [
            [0.16, 0.6, -0.76, 0.87, -0.5, -0.5, 1.5],
            [0.16, 0.6, 0.18, 0.87, -0.5, -0.5, -0.5],
            [1.04, 0.29, -0.76, -0.87, -0.5, 1.5, -0.5],
            [-1.36, -1.48, 1.34, -0.87, 1.5, -0.5, -0.5],
        ],
        ("tf", None): [
            [0.2, 0.4, 0.0, 0.2, 0.0, 0.0, 0.2],
            [0.2, 0.4, 0.2, 0.2, 0.0, 0.0, 0.0],
            [0.33, 0.33, 0.0, 0.0, 0.0, 0.33, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
        ],
        ("tf", "minmax"): [
            [0.4, 0.8, 0.0, 0.4, 0.0, 0.0, 0.4],
            [0.4, 0.8, 0.4, 0.4, 0.0, 0.0, 0.0],
            [0.67, 0.67, 0.0, 0.0, 0.0, 0.67, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ],
        ("tf", "std"): [
            [0.12, 0.61, -0.74, 0.87, -0.5, -0.5, 1.5],
            [0.12, 0.61, 0.11, 0.87, -0.5, -0.5, -0.5],
            [1.09, 0.26, -0.74, -0.87, -0.5, 1.5, -0.5],
            [-1.33, -1.48, 1.38, -0.87, 1.5, -0.5, -0.5],
        ],
        ("tfidf", None): [
            [0.27, 0.53, 0.0, 0.4, 0.0, 0.0, 0.8],
            [0.27, 0.53, 0.4, 0.4, 0.0, 0.0, 0.0],
            [0.44, 0.44, 0.0, 0.0, 0.0, 1.33, 0.0],
            [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0],
        ],
        ("tfidf", "minmax"): [
            [0.13, 0.27, 0.0, 0.2, 0.0, 0.0, 0.4],
            [0.13, 0.27, 0.2, 0.2, 0.0, 0.0, 0.0],
            [0.22, 0.22, 0.0, 0.0, 0.0, 0.67, 0.0],
            [0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
        ],
        ("tfidf", "std"): [
            [0.12, 0.61, -0.74, 0.87, -0.5, -0.5, 1.5],
            [0.12, 0.61, 0.11, 0.87, -0.5, -0.5, -0.5],
            [1.09, 0.26, -0.74, -0.87, -0.5, 1.5, -0.5],
            [-1.33, -1.48, 1.38, -0.87, 1.5, -0.5, -0.5],
        ],
    }
