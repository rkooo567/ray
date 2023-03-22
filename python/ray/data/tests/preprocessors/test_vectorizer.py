from collections import Counter

import pandas as pd
import pytest

import ray
from ray.data.preprocessors import CountVectorizer


def test_count_vectorizer():
    """Tests basic CountVectorizer functionality."""

    # Increase data size & repartition to test for
    # discuss.ray.io/t/xgboost-ray-crashes-when-used-for-multiclass-text-classification
    row_multiplier = 100000

    col_a = ["a b b c c c", "a a a a c"] * row_multiplier
    col_b = ["apple", "banana banana banana"] * row_multiplier
    in_df = pd.DataFrame.from_dict({"A": col_a, "B": col_b})
    ds = ray.data.from_pandas(in_df).repartition(10)

    vectorizer = CountVectorizer(["A", "B"])
    vectorizer.fit(ds)
    assert vectorizer.stats_ == {
        "token_counts(A)": Counter(
            {"a": 5 * row_multiplier, "c": 4 * row_multiplier, "b": 2 * row_multiplier}
        ),
        "token_counts(B)": Counter(
            {"banana": 3 * row_multiplier, "apple": 1 * row_multiplier}
        ),
    }

    transformed = vectorizer.transform(ds)
    out_df = transformed.to_pandas(limit=float("inf"))

    processed_col_a_a = [1, 4] * row_multiplier
    processed_col_a_c = [3, 1] * row_multiplier
    processed_col_a_b = [2, 0] * row_multiplier
    processed_col_b_banana = [0, 3] * row_multiplier
    processed_col_b_apple = [1, 0] * row_multiplier

    expected_df = pd.DataFrame.from_dict(
        {
            "A_a": processed_col_a_a,
            "A_c": processed_col_a_c,
            "A_b": processed_col_a_b,
            "B_banana": processed_col_b_banana,
            "B_apple": processed_col_b_apple,
        }
    )

    assert out_df.equals(expected_df)

    # max_features
    vectorizer = CountVectorizer(["A", "B"], max_features=2)
    vectorizer.fit(ds)

    assert vectorizer.stats_ == {
        "token_counts(A)": Counter({"a": 5 * row_multiplier, "c": 4 * row_multiplier}),
        "token_counts(B)": Counter(
            {"banana": 3 * row_multiplier, "apple": 1 * row_multiplier}
        ),
    }

    transformed = vectorizer.transform(ds)
    out_df = transformed.to_pandas(limit=float("inf"))

    processed_col_a_a = [1, 4] * row_multiplier
    processed_col_a_c = [3, 1] * row_multiplier
    processed_col_b_banana = [0, 3] * row_multiplier
    processed_col_b_apple = [1, 0] * row_multiplier

    expected_df = pd.DataFrame.from_dict(
        {
            "A_a": processed_col_a_a,
            "A_c": processed_col_a_c,
            "B_banana": processed_col_b_banana,
            "B_apple": processed_col_b_apple,
        }
    )

    assert out_df.equals(expected_df)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
