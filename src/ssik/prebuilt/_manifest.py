"""Loader for ``MANIFEST.toml`` — the prebuilt-arm metadata source of truth.

Reads the sibling ``MANIFEST.toml`` and exposes per-arm metadata as a
typed dataclass. All consumers (doc generators, test parametrisations,
the build orchestration in ``scripts/regen_artifacts.py``, the
``examples/04_compare_vs_eaik.py`` bench, the ``ssik add-arm`` CLI)
read from this loader rather than hard-coding arm lists.

See ``MANIFEST.toml`` itself for the schema reference.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

_MANIFEST_PATH = Path(__file__).resolve().parent / "MANIFEST.toml"

__all__ = ["MANIFEST_PATH", "Arm", "ArmBench", "ArmKnownGap", "load_manifest"]

MANIFEST_PATH = _MANIFEST_PATH


@dataclass(frozen=True)
class ArmBench:
    """Latest bench measurements from ``examples/04_compare_vs_eaik.py``."""

    ms_mean: float
    ms_ci95: float
    max_fk: float
    sols_min: int
    sols_max: int


@dataclass(frozen=True)
class ArmKnownGap:
    """A known coverage gap that's xfailed in the uniform-fuzz test."""

    xfail_reason: str


@dataclass(frozen=True)
class Arm:
    """One prebuilt-arm entry in ``MANIFEST.toml``.

    Field semantics: see ``MANIFEST.toml``'s top-of-file schema comment.
    """

    name: str  # e.g. "rizon10_ik"
    display_name: str
    short_name: str
    table_name: str  # compact form for README / quickstart prebuilt tables
    fixture: str
    fixture_kind: Literal["urdf", "specs"]
    base_link: str
    ee_link: str
    dof: int
    solver: str
    tier: int
    kinematic_class: str
    short_class: str
    class_tags: tuple[str, ...]
    slow_build: bool
    build_time_sec: int
    artifact_size_kb: int
    sample_q: tuple[float, ...]
    fk_ceiling_fuzz: float
    platform_drift: bool
    drift_markers: tuple[str, ...] = field(default=())
    specs_fn: str | None = None
    bench: ArmBench | None = None
    known_gaps: ArmKnownGap | None = None


def _coerce_arm(name: str, body: dict[str, object]) -> Arm:
    """Build an :class:`Arm` from a TOML-loaded dict."""
    bench_dict = body.get("bench")
    bench = None
    if isinstance(bench_dict, dict):
        bench = ArmBench(
            ms_mean=float(bench_dict["ms_mean"]),
            ms_ci95=float(bench_dict["ms_ci95"]),
            max_fk=float(bench_dict["max_fk"]),
            sols_min=int(bench_dict["sols_min"]),
            sols_max=int(bench_dict["sols_max"]),
        )
    gaps_dict = body.get("known_gaps")
    known_gaps = None
    if isinstance(gaps_dict, dict):
        known_gaps = ArmKnownGap(xfail_reason=str(gaps_dict["xfail_reason"]))

    fixture_kind = body["fixture_kind"]
    if fixture_kind not in ("urdf", "specs"):
        raise ValueError(
            f"arm {name!r}: fixture_kind must be 'urdf' or 'specs', got {fixture_kind!r}"
        )
    # ``specs_fn`` is required when fixture_kind == "specs"; absent for urdf.
    specs_fn = body.get("specs_fn")
    if fixture_kind == "specs" and not specs_fn:
        raise ValueError(f"arm {name!r}: fixture_kind='specs' requires specs_fn")
    if fixture_kind == "urdf" and specs_fn:
        raise ValueError(f"arm {name!r}: fixture_kind='urdf' but specs_fn is set; remove specs_fn")

    return Arm(
        name=name,
        display_name=str(body["display_name"]),
        short_name=str(body["short_name"]),
        # ``table_name`` defaults to display_name when omitted from the
        # manifest entry. Override per-arm when display_name carries a
        # parenthesised model variant (e.g. "Kinova JACO 2 (j2n6s200)")
        # that doesn't read well in compact table cells.
        table_name=str(body.get("table_name", body["display_name"])),
        fixture=str(body["fixture"]),
        fixture_kind=fixture_kind,  # type: ignore[arg-type]
        specs_fn=str(specs_fn) if specs_fn else None,
        base_link=str(body["base_link"]),
        ee_link=str(body["ee_link"]),
        dof=int(body["dof"]),
        solver=str(body["solver"]),
        tier=int(body["tier"]),
        kinematic_class=str(body["kinematic_class"]),
        short_class=str(body["short_class"]),
        class_tags=tuple(str(t) for t in body["class_tags"]),  # type: ignore[arg-type]
        slow_build=bool(body["slow_build"]),
        build_time_sec=int(body["build_time_sec"]),
        artifact_size_kb=int(body["artifact_size_kb"]),
        sample_q=tuple(float(q) for q in body["sample_q"]),  # type: ignore[arg-type]
        fk_ceiling_fuzz=float(body["fk_ceiling_fuzz"]),
        platform_drift=bool(body["platform_drift"]),
        drift_markers=tuple(str(m) for m in body.get("drift_markers", [])),  # type: ignore[arg-type]
        bench=bench,
        known_gaps=known_gaps,
    )


@lru_cache(maxsize=1)
def load_manifest(path: Path | None = None) -> dict[str, Arm]:
    """Parse ``MANIFEST.toml`` and return ``{arm_name: Arm, ...}`` in
    declaration order.

    Order is preserved (Python dict iteration order matches TOML
    file order), so downstream consumers that want a stable arm ordering
    in tables / lists can just iterate the returned dict.

    Result is cached: subsequent calls in the same process return the
    same dict. Pass an explicit ``path`` for tests that need to load
    an alternate manifest (which also bypasses the cache).
    """
    target = path if path is not None else _MANIFEST_PATH
    if path is not None:
        # Bypass the cache for explicit-path loads (test override).
        return _load_uncached(target)
    return _load_uncached(target)


def _load_uncached(path: Path) -> dict[str, Arm]:
    with path.open("rb") as fp:
        data = tomllib.load(fp)
    raw_arms = data.get("arms")
    if not isinstance(raw_arms, dict):
        raise ValueError(f"{path}: top-level 'arms' table is missing or malformed")
    return {name: _coerce_arm(name, body) for name, body in raw_arms.items()}
