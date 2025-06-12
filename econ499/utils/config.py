from pathlib import Path
import yaml

__all__ = ["load_config"]

def load_config(path: str | Path) -> dict:
    """Load a YAML config file relative to project root and expand relative path entries.

    Any string value contained in the top-level ``paths`` mapping of the YAML
    file will be converted to an absolute path **relative to the detected
    project root**. This ensures that CLI scripts can be run from arbitrary
    working directories without breaking inter-module path references.
    """
    project_root = Path(__file__).resolve().parents[2]  # econ499/utils/ -> econ499/ -> project root
    cfg_path = project_root / Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Expand relative paths so that downstream modules always deal with
    # absolute paths. Only mutate entries under the ``paths`` section and only
    # if the value is a str that represents a *relative* filesystem path.
    if isinstance(cfg, dict) and "paths" in cfg:
        for k, v in cfg["paths"].items():
            if isinstance(v, str) and not Path(v).is_absolute():
                cfg["paths"][k] = str((project_root / Path(v)).resolve())

    return cfg 