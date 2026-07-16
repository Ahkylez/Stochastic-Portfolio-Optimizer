"""Locate the compiled optimizer binary in a platform-aware way.

The build layout differs by toolchain:
  * Windows (MSVC, multi-config generator):  build/Release/optimizer_app.exe
  * Linux / macOS (single-config generator):  build/optimizer_app

Call ``optimizer_exe_path()`` instead of hard-coding either layout so the same
Python entry points run on Windows and WSL/Linux without edits.
"""

import os
import sys

# repo root == parent of this file's directory (src/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def optimizer_exe_path(root: str | None = None) -> str:
    """Return the path to the ``optimizer_app`` executable.

    Args:
        root: repository root that contains ``build/``. Defaults to this repo.

    Returns the first existing candidate for the current OS; if none exists yet
    (not built), returns the conventional path so the caller's error message
    points at the expected location.
    """
    root = root or _REPO_ROOT
    build = os.path.join(root, "build")
    if sys.platform.startswith("win"):
        candidates = [
            os.path.join(build, "Release", "optimizer_app.exe"),
            os.path.join(build, "optimizer_app.exe"),
        ]
    else:
        candidates = [
            os.path.join(build, "optimizer_app"),
            os.path.join(build, "Release", "optimizer_app"),
        ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]
