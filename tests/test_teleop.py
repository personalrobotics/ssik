"""Teleoperation frame helpers (``ssik.teleop``) and the ``06_teleop`` example.

The helpers are rigid-transform algebra, so the tests check the algebra's
invariants on random rigid poses rather than particular outputs: calibration
and tool offsets round-trip, compose with FK and IK the way the docs say, and
a clutch reproduces the device's relative motion rigidly, without a jump at
engagement, with scaling equal to ``scale_about`` at the device anchor.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

import ssik
from ssik.chart import se3_exp
from ssik.teleop import (
    Clutch,
    apply_calibration,
    calibration_from,
    flange_to_tcp,
    invert,
    scale_about,
    tcp_to_flange,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
TOL = 1e-12


def _poses(n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [se3_exp(rng.normal(scale=[0.5, 0.5, 0.5, 1.0, 1.0, 1.0])) for _ in range(n)]


def _close(a: np.ndarray, b: np.ndarray) -> None:
    np.testing.assert_allclose(a, b, atol=TOL)


@pytest.mark.parametrize("seed", range(5))
def test_calibration_and_tool_round_trip(seed: int) -> None:
    D, R, D2, tool = _poses(4, seed)
    _close(invert(D) @ D, np.eye(4))
    C = calibration_from(D, R)
    # The calibration maps the reading it was taken from onto the robot pose,
    # and maps every other reading rigidly: relative motion is unchanged.
    _close(apply_calibration(C, D), R)
    _close(invert(apply_calibration(C, D)) @ apply_calibration(C, D2), invert(D) @ D2)
    # The tool offset and its inverse undo each other in both orders.
    _close(flange_to_tcp(tcp_to_flange(R, tool), tool), R)
    _close(tcp_to_flange(flange_to_tcp(R, tool), tool), R)


def test_tool_offset_composes_with_ik() -> None:
    """Solving for ``tcp_to_flange(T_tcp)`` puts the TCP at ``T_tcp``."""
    arm = ssik.Manipulator.from_prebuilt("ur5e")
    tool = _poses(1, 7)[0]
    tool[:3, 3] *= 0.1  # a 10 cm-scale tool
    q = np.array([0.2, -1.3, 1.4, -1.2, -1.5, 0.3])
    T_tcp = flange_to_tcp(arm.fk(q), tool)
    sols = arm.solve(tcp_to_flange(T_tcp, tool), q_seed=q, max_solutions=1)
    np.testing.assert_allclose(sols[0].q, q, atol=1e-9)
    np.testing.assert_allclose(flange_to_tcp(arm.fk(sols[0].q), tool), T_tcp, atol=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_scale_about(seed: int) -> None:
    T, A = _poses(2, seed)
    a = A[:3, 3]
    S = scale_about(T, 0.25, A)
    _close(S[:3, :3], T[:3, :3])  # rotation never scaled
    _close(S[:3, 3] - a, 0.25 * (T[:3, 3] - a))  # positions scale about the anchor
    _close(scale_about(A, 0.25, a), A)  # the anchor is fixed
    _close(scale_about(T, 1.0, a), T)
    _close(scale_about(scale_about(T, 0.5, a), 3.0, a), scale_about(T, 1.5, a))


def _engaged(clutch: Clutch, D: np.ndarray) -> np.ndarray:
    T = clutch.target(D)
    assert T is not None
    return T


@pytest.mark.parametrize("frame", ["world", "tool"])
@pytest.mark.parametrize("seed", range(5))
def test_clutch_common_invariants(seed: int, frame: Literal["world", "tool"]) -> None:
    A_d, A_r, D1, D2, B_d = _poses(5, seed)
    clutch = Clutch(frame=frame)
    assert not clutch.engaged
    assert clutch.target(D1) is None
    clutch.engage(A_d, A_r)
    # No jump at engagement.
    _close(_engaged(clutch, A_d), A_r)
    # Composition: releasing and engaging again where the device and arm are
    # now continues exactly as if the clutch had stayed engaged.
    T1 = _engaged(clutch, D1)
    T2 = _engaged(clutch, D2)
    clutch.release()
    assert clutch.target(D2) is None
    clutch.engage(D1, T1)
    _close(_engaged(clutch, D2), T2)
    # Indexing: engaged at a moved device, it continues from the arm's pose.
    clutch.engage(B_d, T1)
    _close(_engaged(clutch, B_d), T1)

    # Scale acts on translation only, and equals scale_about at the device
    # anchor followed by the unscaled clutch.
    scaled = Clutch(scale=0.3, frame=frame)
    scaled.engage(A_d, A_r)
    plain = Clutch(frame=frame)
    plain.engage(A_d, A_r)
    _close(_engaged(scaled, D1)[:3, :3], _engaged(plain, D1)[:3, :3])
    _close(_engaged(scaled, D1), _engaged(plain, scale_about(D1, 0.3, A_d)))
    _close(_engaged(scaled, A_d), A_r)


@pytest.mark.parametrize("seed", range(5))
def test_clutch_world_frame(seed: int) -> None:
    A_d, A_r, D1, D2, _ = _poses(5, seed)
    clutch = Clutch(scale=0.4)  # world is the default
    assert clutch.frame == "world"
    clutch.engage(A_d, A_r)
    # A pure device translation moves the arm by scale times it, along the
    # same base-frame direction, whatever the device and tool orientations.
    delta = np.array([0.1, -0.2, 0.05])
    D = A_d.copy()
    D[:3, 3] += delta
    T = _engaged(clutch, D)
    _close(T[:3, 3] - A_r[:3, 3], 0.4 * delta)
    _close(T[:3, :3], A_r[:3, :3])
    # Rigidity in the base frame: between two device poses the arm turns by
    # the device's base-frame rotation and moves by scale times its
    # translation.
    T1, T2 = _engaged(clutch, D1), _engaged(clutch, D2)
    _close(T2[:3, :3] @ T1[:3, :3].T, D2[:3, :3] @ D1[:3, :3].T)
    _close(T2[:3, 3] - T1[:3, 3], 0.4 * (D2[:3, 3] - D1[:3, 3]))


@pytest.mark.parametrize("seed", range(5))
def test_clutch_tool_frame(seed: int) -> None:
    A_d, A_r, D1, D2, _ = _poses(5, seed)
    clutch = Clutch(frame="tool")
    clutch.engage(A_d, A_r)
    # With scale 1 the device's path is reproduced rigidly: relative motion
    # in the moving frame is preserved exactly.
    _close(invert(_engaged(clutch, D1)) @ _engaged(clutch, D2), invert(D1) @ D2)
    # A device translation along its own axes moves the tool along the tool's
    # axes as they were at engagement.
    step = np.eye(4)
    step[:3, 3] = [0.1, -0.2, 0.05]
    _close(_engaged(clutch, A_d @ step), A_r @ step)
    # The two modes agree when the anchors have the same orientation.
    B_r = A_r.copy()
    B_r[:3, :3] = A_d[:3, :3]
    world = Clutch()
    world.engage(A_d, B_r)
    clutch.engage(A_d, B_r)
    _close(_engaged(world, D1), _engaged(clutch, D1))


def test_frame_inputs_follow_the_solve_contract() -> None:
    T = _poses(1, 0)[0]
    bad = T.copy()
    bad[:3, :3] *= 1.1  # not a rotation
    with pytest.raises(ValueError, match="T_device"):
        apply_calibration(T, bad)
    with pytest.raises(TypeError, match="T_robot"):
        Clutch().engage(T, T.astype(complex))
    with pytest.raises(ValueError, match="scale must be finite and > 0"):
        Clutch(scale=0.0)
    with pytest.raises(ValueError, match="frame must be 'world' or 'tool'"):
        Clutch(frame="base")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="scale must be a real number"):
        scale_about(T, True, T)
    with pytest.raises(ValueError, match="anchor must be"):
        scale_about(T, 2.0, np.zeros(4))


def test_teleop_example_runs_headless() -> None:
    """``examples/06_teleop.py`` runs end to end with its scripted source and
    shows the statuses a teleop loop meets: tracking, a speed limit, an
    unreachable stretch it holds through, and recovery."""
    out = subprocess.run(
        [sys.executable, str(EXAMPLES / "06_teleop.py")],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    ).stdout
    lines = out.splitlines()
    statuses = [ln.split()[1] for ln in lines if ln.startswith("t=")]
    assert statuses[0] == "OK"
    assert "LIMITED" in statuses
    assert "HELD" in statuses
    assert "(unreachable)" in out
    assert statuses[-1] == "OK"
    assert lines[-1].startswith("final q:")


@pytest.mark.parametrize(
    "mode",
    [
        ["--tour", "--tour-per-arm", "0.2"],
        ["--self-motion", "--self-motion-seconds", "0.3"],
    ],
    ids=["tour", "self_motion"],
)
def test_viser_demo_runs_headless(mode: list[str]) -> None:
    """``examples/05_viser_interactive_ik.py`` (rebuilt on ``Tracker``) runs its
    scripted modes end to end with primitives, no browser and no downloads.
    Needs ``viser`` (``pip install 'ssik[demo]'``)."""
    pytest.importorskip("viser")
    out = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "05_viser_interactive_ik.py"),
            "--no-meshes",
            "--tour-exit",
            "--tour-delay",
            "0",
            "--tour-settle",
            "0",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            *mode,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    ).stdout
    if mode[0] == "--tour":
        # Every arm of the tour loaded and was driven, then the tour ended.
        assert out.count("  tour: ") == 9
        assert "tour: complete" in out
    else:
        assert "EE drift <=" in out
        assert "self-motion: complete" in out
