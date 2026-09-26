"""Guards for the vendor-subpackage prebuilt namespace (#421).

The reorg is contractually **non-breaking** and **lazy**:

- every legacy flat name resolves and is the *same module object* as its
  hierarchical vendor path (via ``from``, ``importlib``, and attribute access);
- ``import ssik`` / ``import ssik.prebuilt`` / ``import ssik.prebuilt.<vendor>``
  import **zero** arm artifacts;
- ``ssik.list_arms()`` returns the whole catalog without importing any artifact.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import ssik.prebuilt as prebuilt
from ssik.prebuilt._manifest import load_manifest

_ARMS = load_manifest()


def test_every_alias_resolves_and_matches_hierarchical() -> None:
    """Each flat alias == its ``<vendor>.<basename>`` module, via all forms."""
    for arm in _ARMS.values():
        flat = importlib.import_module(f"ssik.prebuilt.{arm.name}")
        hier = importlib.import_module(arm.hier_module)
        attr = getattr(prebuilt, arm.name)
        assert flat is hier is attr, f"{arm.name}: flat/hier/attr are not the same module"
        assert hasattr(flat, "solve"), f"{arm.name}: missing solve"
        assert hasattr(flat, "fk"), f"{arm.name}: missing fk"


def test_alias_map_matches_manifest() -> None:
    """The generated ``_LEGACY_ALIASES`` covers exactly the manifest arms."""
    expected = {a.name: f"{a.vendor}.{a.module_basename}" for a in _ARMS.values()}
    assert expected == prebuilt._LEGACY_ALIASES


def test_list_arms_catalog() -> None:
    cat = {a.name for a in prebuilt.list_arms()}
    assert cat == set(_ARMS)
    # filtering by vendor
    ur = prebuilt.list_arms(vendor="universal_robots")
    assert ur
    assert all(a.vendor == "universal_robots" for a in ur)


def test_dir_exposes_aliases_and_vendors() -> None:
    d = set(dir(prebuilt))
    assert "list_arms" in d
    assert "ur5_ik" in d  # a legacy flat alias
    assert "universal_robots" in d  # vendor packages
    assert "fanuc" in d


@pytest.mark.parametrize(
    "snippet",
    [
        # import ssik: no arm artifact loaded
        "import ssik; import sys; "
        "n=sum(1 for m in sys.modules if m.startswith('ssik.prebuilt.') and m.endswith('_ik')); "
        "assert n==0, n",
        # list_arms(): catalog without importing any artifact
        "import ssik, sys; ssik.list_arms(); "
        "n=sum(1 for m in sys.modules if m.startswith('ssik.prebuilt.') and m.endswith('_ik')); "
        "assert n==0, n",
        # importing a vendor package: no arm loaded
        "import ssik.prebuilt.fanuc, sys; "
        "n=sum(1 for m in sys.modules if m.startswith('ssik.prebuilt.') and m.endswith('_ik')); "
        "assert n==0, n",
        # importing one arm loads exactly that arm
        "import ssik.prebuilt.universal_robots.ur5_ik, sys; "
        "n=sum(1 for m in sys.modules if m.startswith('ssik.prebuilt.') and m.endswith('_ik')); "
        "assert n==1, n",
        # from_prebuilt(): resolving by name loads exactly the arm asked for (#587)
        "import ssik, sys; ssik.Manipulator.from_prebuilt('panda'); "
        "loaded=[m for m in sys.modules if m.startswith('ssik.prebuilt.') and m.endswith('_ik')]; "
        "assert loaded==['ssik.prebuilt.franka.panda_ik'], loaded",
    ],
)
def test_laziness_in_fresh_interpreter(snippet: str) -> None:
    """Laziness must hold in a clean process (in-process sys.modules is polluted
    by other tests)."""
    r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
    assert r.returncode == 0, f"laziness violated:\n{r.stderr}"


# The parameter set ``Manipulator.solve`` forwards when it delegates to an
# artifact (#587). Hard-coded there rather than inspected per call, so it is
# pinned here instead.
_ARTIFACT_SOLVE_PARAMS = frozenset(
    {
        "T_target",
        "max_solutions",
        "q_seed",
        "respect_limits",
        "allow_refinement",
        "allow_rescue",
        "policy",
        "refinement_max_iters",
        "seed_metric",
        "seed_tolerance",
        "enumerate_windings",
        "native",
    }
)


def test_every_artifact_takes_the_delegated_solve_signature() -> None:
    """``Manipulator.from_prebuilt(...).solve`` forwards a fixed keyword set to
    the artifact. Codegen emits one ``solve`` signature for all 72 arms, so a
    change there would break delegation at runtime on whichever arm a user
    happened to pick; catch it here instead.
    """
    import inspect

    offenders = {}
    for name, arm in load_manifest().items():
        params = frozenset(
            inspect.signature(importlib.import_module(arm.hier_module).solve).parameters
        )
        if params != _ARTIFACT_SOLVE_PARAMS:
            offenders[name] = sorted(params ^ _ARTIFACT_SOLVE_PARAMS)
    assert not offenders, f"artifact solve() signatures drifted from the delegated set: {offenders}"
