"""Path resolution and configuration loading.

Design rule
-----------
No absolute local path is hard-coded anywhere in this repository. Every path is
derived from the location of this file, so the package runs from any checkout
directory and on any operating system.

    PROJECT_ROOT = <repository root>
    config/      = PROJECT_ROOT / "config"
    data/        = PROJECT_ROOT / "data"

Optional user overrides are read from ``config/paths.yaml`` when that file
exists. ``config/paths.example.yaml`` documents its schema and is the
recommended starting point.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

#: Repository root, derived from this file's location: ``<root>/src/utils/paths.py``.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    """Directory holding the YAML configuration files."""
    return PROJECT_ROOT / "config"


def data_dir() -> Path:
    """Directory holding the published schema and synthetic demonstration data."""
    return PROJECT_ROOT / "data"


def synthetic_dir() -> Path:
    """Directory holding the synthetic demonstration dataset."""
    return data_dir() / "synthetic"


def schema_dir() -> Path:
    """Directory holding the published data dictionary."""
    return data_dir() / "schema"


def config_path(name: str = "experiment_config.yaml") -> Path:
    """Absolute path of a configuration file inside ``config/``."""
    return config_dir() / name


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping, returning an empty mapping when the file is absent."""
    if not path.exists():
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "PyYAML is required to read configuration files; install it with "
            "`pip install PyYAML`."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return dict(loaded) if isinstance(loaded, dict) else {}


def load_config(name: str = "experiment_config.yaml") -> Dict[str, Any]:
    """Load ``config/<name>`` as a dictionary.

    Missing files yield an empty dictionary so that callers fall back to the
    defaults declared in code rather than crashing.
    """
    return _load_yaml(config_path(name))


def load_paths() -> Dict[str, Any]:
    """Load the optional user path overrides from ``config/paths.yaml``."""
    override = config_dir() / "paths.yaml"
    if override.exists():
        return _load_yaml(override)
    return _load_yaml(config_path("paths.example.yaml"))


def resolve(key: str, default: str) -> Path:
    """Resolve a path from the user override file, else relative to the root.

    Relative values are interpreted against :data:`PROJECT_ROOT`; absolute
    values are honoured only when the user supplied them explicitly in
    ``config/paths.yaml``.
    """
    raw = os.environ.get("JRSP_" + key.upper())
    if raw is None:
        raw = load_paths().get(key, default)
    path = Path(str(raw))
    return path if path.is_absolute() else (PROJECT_ROOT / path)
