# Copyright (c) AlphaBetter. All rights reserved.
import re
from pathlib import Path
from typing import Dict, Union
import yaml

__all__ = [
    "load_yaml", "increment_name",
]


def load_yaml(file_path: Union[Path, str]) -> Dict:
    file_path = Path(file_path)
    assert file_path.suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file_path} with yaml_load()"

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    if not data.isprintable():
        data = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", data)

    data = yaml.safe_load(data) or {}

    return data


def increment_name(path: Union[Path, str]) -> Path:
    """Increase the save directory"s id if the path already exists.

    Args:
        path (Union[Path, str]): The path to the directory.

    Returns:
        Path: The updated path with an incremented id if the original path already exists.
    """
    # Convert the path to a Path object
    if isinstance(path, str):
        path = Path(path)
    separator = ""

    # If the path already exists, increment the id
    if path.exists():
        # If the path is a file, remove the suffix
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        new_path = path
        for number in range(1, 9999):
            new_path = f"{path}{separator}{number}{suffix}"
            if not Path(new_path).exists():
                break
        path = Path(new_path)

    return path
