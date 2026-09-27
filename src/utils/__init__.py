"""Shared utilities: path resolution, configuration loading, runtime setup.

These helpers exist so that the stage scripts stay thin and cannot drift apart. Path
resolution is centralised so that no script embeds a location, and the deterministic
environment is applied in one place so that no script can forget it.
"""

from .cli import (
    add_common_arguments,
    banner,
    default_case_table,
    default_output_dir,
    demonstration_defaults,
    display_path,
    resolve_output_file,
)
from .env import DETERMINISTIC_ENVIRONMENT, configure_determinism
from .paired import describe_pair, paired_units, per_item_slices
from .paths import (
    PROJECT_ROOT,
    config_path,
    load_config,
    load_paths,
    resolve,
    synthetic_dir,
)

__all__ = [
    # paths
    "PROJECT_ROOT",
    "config_path",
    "load_config",
    "load_paths",
    "resolve",
    "synthetic_dir",
    # runtime
    "DETERMINISTIC_ENVIRONMENT",
    "configure_determinism",
    # command line
    "add_common_arguments",
    "banner",
    "default_case_table",
    "default_output_dir",
    "demonstration_defaults",
    "display_path",
    "resolve_output_file",
    # paired comparison
    "describe_pair",
    "paired_units",
    "per_item_slices",
]
