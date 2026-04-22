"""Smoke tests for the vendored ikfast generator on modern sympy.

These tests verify that `ikfast.py` and `ikfast_generator_cpp.py` import cleanly
and that key public symbols are intact. They do not exercise the symbolic
solver itself — that requires a kinbody shim, which lands in #5.
"""

from __future__ import annotations

EXPECTED_IKTYPES: dict[str, int] = {
    "Transform6D": 0x67000001,
    "Rotation3D": 0x34000002,
    "Translation3D": 0x33000003,
    "Direction3D": 0x23000004,
    "Ray4D": 0x46000005,
    "Lookat3D": 0x23000006,
    "TranslationDirection5D": 0x56000007,
    "TranslationXY2D": 0x22000008,
    "TranslationXYOrientation3D": 0x33000009,
    "TranslationLocalGlobal6D": 0x3600000A,
    "TranslationXAxisAngle4D": 0x4400000B,
    "TranslationYAxisAngle4D": 0x4400000C,
    "TranslationZAxisAngle4D": 0x4400000D,
    "TranslationXAxisAngleZNorm4D": 0x4400000E,
    "TranslationYAxisAngleXNorm4D": 0x4400000F,
    "TranslationZAxisAngleYNorm4D": 0x44000010,
}


def test_vendored_modules_import() -> None:
    from ssik._vendor import ikfast, ikfast_generator_cpp  # noqa: F401


def test_iksolver_class_present() -> None:
    from ssik._vendor.ikfast import IKFastSolver

    assert callable(IKFastSolver)
    assert hasattr(IKFastSolver, "GetSolvers")
    assert hasattr(IKFastSolver, "generateIkSolver")


def test_solver_dispatch_table_complete() -> None:
    from ssik._vendor.ikfast import IKFastSolver

    solvers = IKFastSolver.GetSolvers()
    expected_lower_keys = {
        "transform6d",
        "rotation3d",
        "translation3d",
        "direction3d",
        "ray4d",
        "lookat3d",
        "translationdirection5d",
        "translationxy2d",
        "translationxyorientation3d",
        "translationxaxisangle4d",
        "translationyaxisangle4d",
        "translationzaxisangle4d",
        "translationxaxisangleznorm4d",
        "translationyaxisanglexnorm4d",
        "translationzaxisangleynorm4d",
    }
    assert set(solvers.keys()) == expected_lower_keys
    assert all(callable(fn) for fn in solvers.values())


def test_iktype_constants_intact() -> None:
    from ssik._vendor.ikfast_generator_cpp import IkType

    for name, expected_hex in EXPECTED_IKTYPES.items():
        assert getattr(IkType, name) == expected_hex, (
            f"IkType.{name} = {hex(getattr(IkType, name))}, expected {hex(expected_hex)}"
        )
