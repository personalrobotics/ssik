"""MJCF → KinBody adapter (#343): FK parity vs mujoco + edge cases.

Gated on the optional ``mujoco`` dependency (the ``mjcf`` extra; also in the dev
group so CI runs these). The gold-standard oracle is mujoco's own ``mj_forward``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from ssik._mjcf import load_mjcf_kinbody_normalized  # noqa: E402
from ssik.kinematics.poe_fk import poe_forward_kinematics  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
TOY = FIXTURES / "toy3r.xml"


def _mujoco_base_ee_fk(path: Path, base: str, ee: str, q: np.ndarray) -> np.ndarray:
    """Reference base→ee transform from mujoco at joint config ``q``."""
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    for jid in range(m.njnt):
        d.qpos[int(m.jnt_qposadr[jid])] = q[jid]
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base)
    eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee)
    t_base = np.eye(4)
    t_base[:3, :3] = d.xmat[bid].reshape(3, 3)
    t_base[:3, 3] = d.xpos[bid]
    t_ee = np.eye(4)
    t_ee[:3, :3] = d.xmat[eid].reshape(3, 3)
    t_ee[:3, 3] = d.xpos[eid]
    return np.linalg.inv(t_base) @ t_ee


def test_mjcf_fk_matches_mujoco() -> None:
    """The POE-normalized KinBody's FK matches mujoco's mj_forward at random q."""
    kb = load_mjcf_kinbody_normalized(TOY, "base", "link3")
    assert len(kb.joints) == 3
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        q = rng.uniform(-2.0, 2.0, size=3)
        ref = _mujoco_base_ee_fk(TOY, "base", "link3", q)
        got = poe_forward_kinematics(kb, q)
        worst = max(worst, float(np.abs(got - ref).max()))
    assert worst < 1e-12, f"MJCF KinBody FK off from mujoco by {worst:.2e}"


def test_mjcf_limits_are_read() -> None:
    kb = load_mjcf_kinbody_normalized(TOY, "base", "link3")
    limits = [j.limits for j in kb.joints]
    assert limits[0] is None  # j1 unlimited
    assert limits[1] == pytest.approx((-2.0, 2.0))  # j2 range
    assert limits[2] is None  # j3 unlimited


def test_mjcf_missing_body_raises() -> None:
    with pytest.raises(ValueError, match="no body named"):
        load_mjcf_kinbody_normalized(TOY, "base", "nonexistent")


def test_mjcf_rejects_ball_joint(tmp_path: Path) -> None:
    """Ball/free joints are multi-DOF and unsupported -- raise, don't mis-parse."""
    ball = tmp_path / "ball.xml"
    ball.write_text(
        "<mujoco model='b'><worldbody><body name='base'>"
        "<body name='ee' pos='0 0 0.2'>"
        "<joint name='jb' type='ball'/>"
        "<inertial pos='0 0 0' mass='1' diaginertia='0.01 0.01 0.01'/>"
        "</body></body></worldbody></mujoco>"
    )
    with pytest.raises(NotImplementedError, match="hinge/slide"):
        load_mjcf_kinbody_normalized(ball, "base", "ee")
