"""Deprecation shim: the 'ikfastpy' package has been renamed to 'ssik'.

This module re-exports the public surface of :mod:`ssik` and emits a
``DeprecationWarning`` on import. It exists for one release to give downstream
code a migration window; after that it will be removed and ``import ikfastpy``
will fail.

Update your imports::

    - import ikfastpy
    + import ssik
"""

from __future__ import annotations

import warnings

warnings.warn(
    "The 'ikfastpy' package has been renamed to 'ssik'. "
    "Update your imports: 'import ssik' instead of 'import ikfastpy'. "
    "This shim will be removed in the next release.",
    DeprecationWarning,
    stacklevel=2,
)

from ssik import *  # noqa: E402,F403
from ssik import __version__  # noqa: E402,F401
