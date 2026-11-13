"""YAML configuration loading and small helpers to read / override entries.

The configuration is kept as a plain nested ``dict`` on purpose: it mirrors the
YAML file one-to-one, is trivial to print or dump next to the results, and
needs no extra classes when a new section is added.

Nested entries are addressed with *dotted keys* such as ``"svm.kernel.gamma"``.
The hyper-parameter grids of the cross-validation use the same notation.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = ("seed", "paths", "data", "cost", "svm", "kpca", "cv")


def load_config(path: str | Path) -> dict:
    """Read a YAML configuration file and check that the main sections exist."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a YAML mapping.")
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"{path} is missing the required section(s): {missing}.")
    return config


def get_nested(config: Mapping, dotted_key: str) -> Any:
    """Return ``config[a][b][c]`` for ``dotted_key == "a.b.c"``."""
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(f"Configuration has no entry {dotted_key!r} (failed at {part!r}).")
        node = node[part]
    return node


def set_nested(config: dict, dotted_key: str, value: Any) -> None:
    """Set ``config[a][b][c] = value`` in place, creating intermediate dicts."""
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def with_overrides(config: Mapping, overrides: Mapping[str, Any] | None) -> dict:
    """Return a deep copy of ``config`` with dotted-key ``overrides`` applied."""
    new_config = copy.deepcopy(dict(config))
    for dotted_key, value in (overrides or {}).items():
        set_nested(new_config, dotted_key, copy.deepcopy(value))
    return new_config
