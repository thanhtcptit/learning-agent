from __future__ import annotations

import sys
from pathlib import Path


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_bundle_data_root() -> Path:
    if is_frozen_app():
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            return Path(bundle_root)

    return get_project_root()


def get_runtime_root() -> Path:
    if is_frozen_app():
        return Path(sys.executable).resolve().parent

    return get_project_root()


def get_runtime_file_path(*parts: str) -> Path:
    return get_runtime_root().joinpath(*parts)
