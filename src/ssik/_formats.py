"""Pluggable robot-description format adapters (#470).

``ssik``'s :class:`~ssik._kinbody.KinBody` is the universal internal form;
solvers never see the source format. Ingestion of a concrete format (URDF,
MuJoCo MJCF, and -- in future -- USD/SDF, #83) is captured by one
:class:`FormatAdapter`, and every format-agnostic consumer (``Manipulator``
factories, ``ssik add-arm``, ``regen_artifacts``, the MANIFEST schema) goes
through the registry here instead of branching on the format itself.

Adding a format is therefore: write its loader/vendor functions, build a
:class:`FormatAdapter`, and :func:`register` it. No edits to the CLI, the
artifact regenerator, or the manifest validator.

The two built-in adapters live next to their parsers (``_urdf.py``, ``_mjcf.py``)
and self-register on import; :func:`_ensure_builtins` imports those modules
lazily on first registry access so importing this module stays dependency-free
(``urchin`` / ``mujoco`` are only pulled in when a format is actually used).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from ssik._kinbody import KinBody

__all__ = [
    "FormatAdapter",
    "detect",
    "detect_kind",
    "get",
    "kinds",
    "register",
]


@dataclass(frozen=True)
class FormatAdapter:
    """Everything the format-agnostic pipeline needs for one source format.

    :param kind: the ``fixture_kind`` tag written to MANIFEST.toml and matched by
        :func:`get` (e.g. ``"urdf"``, ``"mjcf"``).
    :param label: human name for CLI messages (e.g. ``"URDF"``).
    :param extensions: source-file suffixes this format claims, lowercase with the
        dot (e.g. ``(".urdf", ".xacro")``). Used by :func:`detect`.
    :param fixture_suffix: extension of the vendored kinematics-only fixture
        (e.g. ``".urdf"``, ``".xml"``).
    :param scaffold_loader_module: module the generated test/artifact imports the
        normalized loader from (e.g. ``"ssik._urdf"``).
    :param scaffold_loader_func: normalized-loader function name in that module
        (e.g. ``"load_urdf_kinbody_normalized"``).
    :param load: ``(path, base, ee, options) -> KinBody`` -- load a source or
        vendored fixture into a POE-normalized KinBody. ``options`` carries
        format-specific knobs (e.g. ``xacro_args`` for URDF); adapters ignore
        keys they don't use.
    :param suggest_base_ee: ``(source_path, options) -> (base, ee, notes)``.
    :param vendor: ``(source_path, dest_path, options) -> (n_a, n_b)`` -- write a
        kinematics-only fixture to ``dest_path`` and return two element counts
        (links/bodies, joints) for the CLI summary.
    :param root_tags: optional XML root element names that positively identify
        this format when the extension is ambiguous (``.xml`` is shared by MJCF
        and others); e.g. ``("robot",)`` for URDF, ``("mujoco",)`` for MJCF.
    """

    kind: str
    label: str
    extensions: tuple[str, ...]
    fixture_suffix: str
    scaffold_loader_module: str
    scaffold_loader_func: str
    load: Callable[[Path, str, str, Mapping[str, Any]], KinBody]
    suggest_base_ee: Callable[[Path, Mapping[str, Any]], tuple[str, str, list[str]]]
    vendor: Callable[[Path, Path, Mapping[str, Any]], tuple[int, int]]
    root_tags: tuple[str, ...] = field(default=())


_REGISTRY: dict[str, FormatAdapter] = {}
_BUILTINS_LOADED = False


def register(adapter: FormatAdapter) -> None:
    """Register a format adapter under its ``kind`` (idempotent overwrite)."""
    _REGISTRY[adapter.kind] = adapter


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Import for side-effect registration. Lazy so this module has no hard
    # dependency on urchin/mujoco; the loaders themselves import those on use.
    _BUILTINS_LOADED = True
    import ssik._mjcf
    import ssik._urdf  # noqa: F401


def kinds() -> tuple[str, ...]:
    """All registered format kinds (sorted)."""
    _ensure_builtins()
    return tuple(sorted(_REGISTRY))


def get(kind: str) -> FormatAdapter:
    """Look up an adapter by its ``fixture_kind``.

    :raises ValueError: if no adapter is registered for ``kind``.
    """
    _ensure_builtins()
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise ValueError(
            f"unknown format kind {kind!r}; registered: {', '.join(kinds())}"
        ) from None


def _sniff_root_tag(path: Path) -> str | None:
    """Local root element name of an XML file, or ``None`` if unreadable."""
    try:
        import xml.etree.ElementTree as ET

        for _event, elem in ET.iterparse(path, events=("start",)):
            tag = elem.tag
            return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else None
    except (OSError, ET.ParseError):
        return None
    return None


def detect(path: str | Path) -> FormatAdapter:
    """Pick the adapter for a source file by extension, disambiguating shared
    extensions (``.xml``) by the XML root element.

    :raises ValueError: if no registered format claims the file.
    """
    _ensure_builtins()
    p = Path(path)
    suffix = p.suffix.lower()

    by_ext = [a for a in _REGISTRY.values() if suffix in a.extensions]
    if len(by_ext) == 1:
        return by_ext[0]
    if len(by_ext) > 1:
        # Ambiguous extension (e.g. .xml): disambiguate by XML root tag.
        root = _sniff_root_tag(p)
        for a in by_ext:
            if root is not None and root in a.root_tags:
                return a
        # Fall back to the first claimant with no root-tag opinion.
        return by_ext[0]

    exts = sorted({e for a in _REGISTRY.values() for e in a.extensions})
    raise ValueError(
        f"no registered format for {p.name!r} (suffix {suffix!r}); "
        f"known extensions: {', '.join(exts)}"
    )


def detect_kind(path: str | Path) -> str:
    """The ``fixture_kind`` of the format that claims ``path``."""
    return detect(path).kind
