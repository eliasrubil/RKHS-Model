"""Data handling: decoding, cache, split and leakage-free preprocessing.

Nothing here touches the network: the download functions are replaced by stubs.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from rkhs_credit import data
from rkhs_credit.data import (
    TabularPreprocessor,
    decode_uci_statlog,
    derive_sensitive_groups,
    load_german_credit,
    normalise_frame,
    split_features_target,
    stratified_split,
)

# Primeras tres filas del archivo original german.data (UCI Statlog).
UCI_SAMPLE = (
    "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1\n"
    "A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2\n"
    "A14 12 A34 A46 2096 A61 A74 2 A93 A101 3 A121 49 A143 A152 1 A172 2 A191 A201 1\n"
)


def test_decode_uci_statlog_uses_openml_names_and_labels():
    frame = decode_uci_statlog(UCI_SAMPLE)
    assert frame.shape == (3, 21)
    assert list(frame.columns[:3]) == ["checking_status", "duration", "credit_history"]
    assert list(frame["class"]) == ["good", "bad", "good"]
    assert list(frame["checking_status"]) == ["<0", "0<=X<200", "no checking"]
    assert list(frame["personal_status"]) == ["male single", "female div/dep/mar", "male single"]
    assert list(frame["credit_amount"]) == [1169, 5951, 2096]
    assert list(frame["foreign_worker"]) == ["yes", "yes", "yes"]


def test_decode_uci_statlog_rejects_unknown_codes():
    with pytest.raises(ValueError, match="Unexpected code"):
        decode_uci_statlog(UCI_SAMPLE.replace("A11", "A19"))


def test_normalise_frame_gives_plain_dtypes():
    frame = pd.DataFrame(
        {
            "amount": pd.Series([1, 2, 3], dtype="int64"),
            "label": pd.Categorical(["'male single'", " none ", "yes"]),
        }
    )
    clean = normalise_frame(frame)
    assert clean["amount"].dtype == np.float64
    assert list(clean["label"]) == ["male single", "none", "yes"]


def _config(tmp_path, fallback_url="https://example.invalid/german.data") -> dict:
    return {
        "paths": {"data_dir": str(tmp_path / "data")},
        "data": {
            "openml_name": "credit-g",
            "openml_version": 1,
            "cache_file": "credit_g.csv",
            "fallback_url": fallback_url,
        },
    }


def test_load_downloads_once_then_reads_the_cache(tmp_path, monkeypatch):
    calls = []

    def fake_openml(name, version, data_dir):
        calls.append((name, version))
        return decode_uci_statlog(UCI_SAMPLE)

    monkeypatch.setattr(data, "_fetch_from_openml", fake_openml)
    config = _config(tmp_path)
    first = load_german_credit(config)
    assert calls == [("credit-g", 1)]
    assert (tmp_path / "data" / "credit_g.csv").exists()
    assert data.data_source(config) == "openml:credit-g:v1"

    def no_network(*args, **kwargs):
        raise AssertionError("the cache must be used instead of downloading again")

    monkeypatch.setattr(data, "_fetch_from_openml", no_network)
    second = load_german_credit(config)
    pd.testing.assert_frame_equal(first, second)
    # "none" es una categoria valida y no debe leerse como valor faltante.
    assert not second.isna().any().any()
    assert "none" in set(second["own_telephone"])


def test_load_falls_back_to_uci_when_openml_fails(tmp_path, monkeypatch):
    def failing_openml(name, version, data_dir):
        raise OSError("network unreachable")

    def fake_urlopen(url, timeout=None):
        return io.BytesIO(UCI_SAMPLE.encode("ascii"))

    monkeypatch.setattr(data, "_fetch_from_openml", failing_openml)
    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    config = _config(tmp_path)
    with pytest.warns(RuntimeWarning, match="Falling back"):
        frame = load_german_credit(config)
    assert frame.shape == (3, 21)
    assert data.data_source(config).startswith("uci:")


def test_load_raises_when_openml_fails_and_fallback_is_disabled(tmp_path, monkeypatch):
    def failing_openml(name, version, data_dir):
        raise OSError("network unreachable")

    monkeypatch.setattr(data, "_fetch_from_openml", failing_openml)
    with pytest.raises(OSError, match="network unreachable"):
        load_german_credit(_config(tmp_path, fallback_url=None))


def test_split_features_target_encodes_bad_as_one():
    frame = decode_uci_statlog(UCI_SAMPLE)
    X, y = split_features_target(frame, target="class", positive_class="bad")
    assert "class" not in X.columns
    assert list(y) == [0, 1, 0]
    with pytest.raises(ValueError):
        split_features_target(frame, target="class", positive_class="unknown")


def test_stratified_split_preserves_the_class_ratio_and_is_reproducible(credit_like_data):
    X, y = credit_like_data
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, seed=3)
    assert len(X_test) == round(0.2 * len(X))
    assert abs(y_train.mean() - y.mean()) < 0.02
    assert abs(y_test.mean() - y.mean()) < 0.02
    assert set(X_train.index).isdisjoint(X_test.index)
    again = stratified_split(X, y, test_size=0.2, seed=3)
    pd.testing.assert_frame_equal(X_test, again[1])


def test_preprocessor_statistics_come_from_the_training_frame_only(credit_like_data):
    X, y = credit_like_data
    X_train, X_test, _, _ = stratified_split(X, y, test_size=0.25, seed=0)
    preprocessor = TabularPreprocessor().fit(X_train)

    numeric = preprocessor.numeric_columns_
    assert numeric == ["duration", "credit_amount", "age"]
    np.testing.assert_allclose(preprocessor.scaler_.mean_, X_train[numeric].mean().to_numpy())

    train_features = preprocessor.transform(X_train)
    np.testing.assert_allclose(train_features[:, :3].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(train_features[:, :3].std(axis=0), 1.0, atol=1e-12)
    # El test se transforma con las estadisticas de train: su media no es cero.
    test_features = preprocessor.transform(X_test)
    assert np.abs(test_features[:, :3].mean(axis=0)).max() > 1e-3
    assert test_features.shape[1] == train_features.shape[1] == len(preprocessor.feature_names_)
    # Cada bloque one-hot suma 1 por fila.
    one_hot = train_features[:, 3:]
    assert set(np.unique(one_hot)) == {0.0, 1.0}
    np.testing.assert_allclose(one_hot.sum(axis=1), len(preprocessor.categorical_columns_))


def test_preprocessor_ignores_unseen_categories_and_can_exclude_columns(credit_like_data):
    X, _ = credit_like_data
    preprocessor = TabularPreprocessor().fit(X)
    unseen = X.iloc[:2].copy()
    unseen["purpose"] = "never seen before"
    features = preprocessor.transform(unseen)
    purpose_columns = [
        i for i, name in enumerate(preprocessor.feature_names_) if name.startswith("purpose=")
    ]
    np.testing.assert_array_equal(features[:, purpose_columns], 0.0)

    sensitive = ["personal_status", "age", "foreign_worker"]
    reduced = TabularPreprocessor(exclude=sensitive).fit(X)
    assert not any(name.split("=")[0] in sensitive for name in reduced.feature_names_)
    assert len(reduced.feature_names_) < len(preprocessor.feature_names_)
    with pytest.raises(RuntimeError):
        TabularPreprocessor().transform(X)


def test_derive_sensitive_groups():
    X = pd.DataFrame(
        {
            "personal_status": ["male single", "female div/dep/mar", "male mar/wid"],
            "age": [25.0, 26.0, 40.0],
            "foreign_worker": ["yes", "no", "yes"],
        }
    )
    groups = derive_sensitive_groups(X, age_threshold=25)
    assert list(groups["sex"]) == ["male", "female", "male"]
    assert list(groups["age_group"]) == ["<=25", ">25", ">25"]
    assert list(groups["foreign_worker"]) == ["yes", "no", "yes"]
