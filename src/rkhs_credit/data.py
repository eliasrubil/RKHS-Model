"""German Credit (Statlog) data: download, local cache, split and preprocessing.

Primary source
    ``sklearn.datasets.fetch_openml("credit-g", version=1)``.

Fallback source
    The original UCI file ``german.data`` (same 1000 rows, same order), decoded
    to the OpenML column names and category labels.  It is only used when
    OpenML cannot be reached and ``data.fallback_url`` is set in the
    configuration.  OpenML has occasional outages and some sandboxed
    environments block it, so the fallback keeps the project runnable.

Either way the result is written once to ``<data_dir>/<cache_file>`` (CSV) and
every later run reads that file, without touching the network.

Target encoding
    ``y = 1`` for the positive class (``"bad"`` credit risk) and ``y = 0`` for
    ``"good"``.  All scores in the project are oriented so that *higher means
    riskier*.
"""

from __future__ import annotations

import io
import json
import urllib.request
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Schema of the UCI Statlog file, in column order.  Categorical attributes map
# the original codes (A11, A12, ...) to the labels used by OpenML ``credit-g``;
# numeric attributes have ``None``.
# ----------------------------------------------------------------------------
UCI_STATLOG_SCHEMA: dict[str, dict[str, str] | None] = {
    "checking_status": {"A11": "<0", "A12": "0<=X<200", "A13": ">=200", "A14": "no checking"},
    "duration": None,
    "credit_history": {
        "A30": "no credits/all paid",
        "A31": "all paid",
        "A32": "existing paid",
        "A33": "delayed previously",
        "A34": "critical/other existing credit",
    },
    "purpose": {
        "A40": "new car",
        "A41": "used car",
        "A42": "furniture/equipment",
        "A43": "radio/tv",
        "A44": "domestic appliance",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "other",
    },
    "credit_amount": None,
    "savings_status": {
        "A61": "<100",
        "A62": "100<=X<500",
        "A63": "500<=X<1000",
        "A64": ">=1000",
        "A65": "no known savings",
    },
    "employment": {
        "A71": "unemployed",
        "A72": "<1",
        "A73": "1<=X<4",
        "A74": "4<=X<7",
        "A75": ">=7",
    },
    "installment_commitment": None,
    "personal_status": {
        "A91": "male div/sep",
        "A92": "female div/dep/mar",
        "A93": "male single",
        "A94": "male mar/wid",
        "A95": "female single",
    },
    "other_parties": {"A101": "none", "A102": "co applicant", "A103": "guarantor"},
    "residence_since": None,
    "property_magnitude": {
        "A121": "real estate",
        "A122": "life insurance",
        "A123": "car",
        "A124": "no known property",
    },
    "age": None,
    "other_payment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "existing_credits": None,
    "job": {
        "A171": "unemp/unskilled non res",
        "A172": "unskilled resident",
        "A173": "skilled",
        "A174": "high qualif/self emp/mgmt",
    },
    "num_dependents": None,
    "own_telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
    "class": {"1": "good", "2": "bad"},
}


# ------------------------------------------------------------------ loading
def load_german_credit(config: Mapping) -> pd.DataFrame:
    """Return the German Credit table (20 attributes + target), using the cache.

    The first call downloads the data and writes the CSV cache; later calls
    only read the cache.
    """
    data_cfg = config["data"]
    data_dir = Path(config["paths"]["data_dir"])
    cache_path = data_dir / data_cfg["cache_file"]
    if cache_path.exists():
        return _read_cache(cache_path)

    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        frame = _fetch_from_openml(data_cfg["openml_name"], data_cfg["openml_version"], data_dir)
        source = f"openml:{data_cfg['openml_name']}:v{data_cfg['openml_version']}"
    except (OSError, ValueError) as error:
        fallback_url = data_cfg.get("fallback_url")
        if not fallback_url:
            raise
        warnings.warn(
            f"OpenML could not be reached ({type(error).__name__}: {error}). "
            f"Falling back to the original UCI file at {fallback_url}.",
            RuntimeWarning,
            stacklevel=2,
        )
        frame = _fetch_from_uci(fallback_url)
        source = f"uci:{fallback_url}"

    frame = normalise_frame(frame)
    frame.to_csv(cache_path, index=False)
    _source_path(cache_path).write_text(json.dumps({"source": source}, indent=2) + "\n")
    # Se relee el CSV para que la primera corrida y las siguientes usen
    # exactamente los mismos tipos de dato.
    return _read_cache(cache_path)


def data_source(config: Mapping) -> str:
    """Where the cached table came from (``"unknown"`` if there is no record)."""
    cache_path = Path(config["paths"]["data_dir"]) / config["data"]["cache_file"]
    source_path = _source_path(cache_path)
    if source_path.exists():
        return json.loads(source_path.read_text())["source"]
    return "unknown"


def _source_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".source.json")


def _read_cache(cache_path: Path) -> pd.DataFrame:
    # keep_default_na=False: la categoria "none" es un valor legitimo, no un faltante.
    return normalise_frame(pd.read_csv(cache_path, keep_default_na=False))


def _fetch_from_openml(name: str, version: int, data_dir: Path) -> pd.DataFrame:
    bunch = fetch_openml(
        name=name,
        version=version,
        as_frame=True,
        parser="auto",
        data_home=str(data_dir / "openml"),
    )
    return bunch.frame


def _fetch_from_uci(url: str, timeout: float = 60.0) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        raw_text = response.read().decode("ascii")
    return decode_uci_statlog(raw_text)


def decode_uci_statlog(raw_text: str) -> pd.DataFrame:
    """Parse the whitespace-separated ``german.data`` file into the OpenML layout."""
    columns = list(UCI_STATLOG_SCHEMA)
    frame = pd.read_csv(io.StringIO(raw_text), sep=r"\s+", header=None, names=columns, dtype=str)
    for column, codes in UCI_STATLOG_SCHEMA.items():
        if codes is None:
            frame[column] = pd.to_numeric(frame[column])
            continue
        unknown = set(frame[column]) - set(codes)
        if unknown:
            raise ValueError(f"Unexpected code(s) {sorted(unknown)} in column {column!r}.")
        frame[column] = frame[column].map(codes)
    return frame


def normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every column a plain dtype: ``float64`` if numeric, ``str`` labels otherwise."""
    frame = frame.copy()
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype(float)
        else:
            # Algunos parsers de ARFF conservan las comillas simples de las etiquetas.
            frame[column] = frame[column].astype(str).str.strip().str.strip("'").astype(object)
    return frame


# ----------------------------------------------------------- target / split
def split_features_target(
    frame: pd.DataFrame, target: str, positive_class: str
) -> tuple[pd.DataFrame, np.ndarray]:
    """Separate the attributes from the binary target (1 = positive class)."""
    labels = set(frame[target].unique())
    if positive_class not in labels or len(labels) != 2:
        raise ValueError(
            f"Target {target!r} must be binary and contain {positive_class!r}; found {labels}."
        )
    y = (frame[target] == positive_class).to_numpy().astype(int)
    X = frame.drop(columns=[target])
    return X, y


def stratified_split(
    X: pd.DataFrame, y: np.ndarray, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Stratified train/test split with a fixed seed (indices are preserved)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def load_and_split(config: Mapping):
    """Convenience wrapper: load the table and return the train/test split."""
    data_cfg = config["data"]
    frame = load_german_credit(config)
    X, y = split_features_target(frame, data_cfg["target"], data_cfg["positive_class"])
    return stratified_split(X, y, data_cfg["test_size"], config["seed"])


# ------------------------------------------------------------ preprocessing
class TabularPreprocessor:
    """Standardise numeric columns and one-hot encode categorical ones.

    The column types are inferred from the dtypes of the frame passed to
    :meth:`fit`.  All statistics (means, standard deviations, category lists)
    come from that frame only, so fitting on the training split and calling
    :meth:`transform` on validation/test data involves no data leakage.

    Parameters
    ----------
    exclude : sequence of str
        Columns to drop before encoding (e.g. the sensitive attributes).
    """

    def __init__(self, exclude: Sequence[str] = ()):
        self.exclude = tuple(exclude)

    def fit(self, X: pd.DataFrame) -> TabularPreprocessor:
        X = self._select(X)
        self.numeric_columns_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_columns_ = [c for c in X.columns if c not in self.numeric_columns_]
        self.scaler_ = StandardScaler().fit(X[self.numeric_columns_].to_numpy(dtype=float))
        # handle_unknown="ignore": una categoria no vista en train se codifica como
        # el vector nulo en lugar de producir un error en validacion/test.
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
            X[self.categorical_columns_].to_numpy(dtype=object)
        )
        one_hot_names = [
            f"{column}={category}"
            for column, categories in zip(
                self.categorical_columns_, self.encoder_.categories_, strict=True
            )
            for category in categories
        ]
        self.feature_names_ = [*self.numeric_columns_, *one_hot_names]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "scaler_"):
            raise RuntimeError("TabularPreprocessor is not fitted yet; call fit first.")
        X = self._select(X)
        numeric = self.scaler_.transform(X[self.numeric_columns_].to_numpy(dtype=float))
        categorical = self.encoder_.transform(X[self.categorical_columns_].to_numpy(dtype=object))
        return np.hstack([numeric, categorical])

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.exclude if c in X.columns])


# ------------------------------------------------------- sensitive groups
def derive_sensitive_groups(X: pd.DataFrame, age_threshold: float) -> pd.DataFrame:
    """Build the group labels used by the subgroup analysis of German Credit.

    * ``sex``: read from the prefix of ``personal_status`` ("male ..." / "female ...").
    * ``age_group``: ``age <= age_threshold`` versus older applicants.
    * ``foreign_worker``: the attribute as given ("yes" / "no").

    The function is specific to this dataset (it knows the three column names
    and how sex is encoded inside ``personal_status``).
    """
    young = f"<={age_threshold:g}"
    older = f">{age_threshold:g}"
    return pd.DataFrame(
        {
            "sex": np.where(
                X["personal_status"].astype(str).str.startswith("female"), "female", "male"
            ),
            "age_group": np.where(X["age"] <= age_threshold, young, older),
            "foreign_worker": X["foreign_worker"].astype(str).to_numpy(),
        },
        index=X.index,
    )
