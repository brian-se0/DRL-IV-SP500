"""Configuration loading utilities."""

from pathlib import Path
import yaml

__all__ = ["load_config", "load_config_from_path"]


def load_config(filename: str) -> dict:
    """Load a YAML config file from the ``config`` directory."""
    root = Path(__file__).resolve().parents[2]
    full_path = root / "config" / filename
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")
    return yaml.safe_load(open(full_path))


def load_config_from_path(path: str | Path) -> dict:
    """Load a YAML config file relative to project root and expand relative path entries."""
    project_root = Path(__file__).resolve().parents[2]  # iv_drl/utils/ -> iv_drl/ -> project root
    cfg_path = project_root / "config" / Path(path)  # Look in config directory
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Expand relative paths so downstream modules always see absolute paths
    if isinstance(cfg, dict) and "paths" in cfg:
        for k, v in cfg["paths"].items():
            if isinstance(v, str) and not Path(v).is_absolute():
                cfg["paths"][k] = str((project_root / Path(v)).resolve())

    return cfg
