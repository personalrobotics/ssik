"""Where the per-arm preparation cache lives, and how entries land in it.

The cache holds per-arm work that costs seconds to minutes and depends only on
the arm's geometry and on ssik's own code: the general-6R Raghavan-Roth
derivations (``ssik.solvers.ikgeo._raghavan_roth``) and the native RR tensor
built from them (``ssik._native``). The user-facing description is
``docs/api.md`` ("Per-arm preparation cache").

This module imports nothing heavy, so a cache hit never pays for sympy.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

ENV_VAR = "SSIK_DERIVATION_CACHE"


def cache_dir() -> Path | None:
    """The cache directory, or ``None`` when ``$SSIK_DERIVATION_CACHE`` is
    ``off`` (or empty)."""
    env = os.environ.get(ENV_VAR)
    if env is not None:
        return None if env.strip().lower() in ("", "off") else Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "ssik" / "rr-derivations"


def source_digest(paths: Iterable[Path]) -> str:
    """SHA-256 over the bytes of each source file, or ``""`` when any is
    unreadable (e.g. a zipapp), in which case callers run without the cache."""
    h = hashlib.sha256()
    for path in paths:
        try:
            h.update(path.read_bytes())
        except OSError:
            return ""
    return h.hexdigest()


def write_atomically(path: Path, blob: bytes) -> None:
    """Write ``blob`` to ``path`` by rename, so a reader never sees a partial
    entry. A read-only or full cache directory is not an error: the caller's
    freshly computed value still stands, it just is not stored."""
    tmp: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        os.replace(tmp, path)
        tmp = None
    except OSError:
        pass
    finally:
        if tmp is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
