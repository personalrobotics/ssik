"""MJCF → KinBody adapter (#343): FK parity vs mujoco + edge cases.

Gated on the optional ``mujoco`` dependency (the ``mjcf`` extra; also in the dev
group so CI runs these). The gold-standard oracle is mujoco's own ``mj_forward``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from ssik._mjcf import (  # noqa: E402
    load_mjcf_kinbody_normalized,
    strip_mjcf_to_fixture,
    suggest_base_ee_mjcf,
)
from ssik.kinematics.poe_fk import poe_forward_kinematics  # noqa: E402
from ssik.manipulator import Manipulator  # noqa: E402

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


def test_from_mjcf_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="MJCF file not found"):
        Manipulator.from_mjcf(FIXTURES / "does_not_exist.xml")


# --- Real MuJoCo Menagerie arms (gated on robot_descriptions) --------------
# The mujoco mj_forward pass is the gold-standard FK oracle: build the KinBody
# via the public auto-detecting entry point, then confirm its FK matches mujoco
# body poses at random q for the *same* chain joints.

robot_descriptions = pytest.importorskip("robot_descriptions")

# (module, expected auto-detected base, expected auto-detected ee, dof). The
# ee assertions pin the gripper-trim (arx_l5 must stop at link7, not a finger).
_REAL_MJCF_ARMS = [
    ("gen3_mj_description", "base_link", "bracelet_link", 7),
    ("iiwa14_mj_description", "base", "link7", 7),
    ("fr3_mj_description", "fr3_link0", "fr3_link7", 7),
    ("arx_l5_mj_description", "base_link", "link7", 7),
]


def _chain_jnt_ids(model: Any, base: str, ee: str) -> list[int]:
    """mujoco joint ids on the base->ee chain, in chain order."""
    hinge, slide = mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base)
    chain: list[int] = []
    while bid != base_id:
        chain.append(bid)
        bid = int(model.body_parentid[bid])
    chain.reverse()
    ids: list[int] = []
    for body in chain:
        j0 = int(model.body_jntadr[body])
        ids += [
            j
            for j in range(j0, j0 + int(model.body_jntnum[body]))
            if int(model.jnt_type[j]) in (hinge, slide)
        ]
    return ids


@pytest.mark.parametrize(
    ("module", "exp_base", "exp_ee"),
    [
        ("viper_mj_description", "base_link", "gripper_link"),
        ("widow_mj_description", "wx250s/base_link", "wx250s/gripper_link"),
    ],
)
def test_suggest_base_ee_keeps_gripper_named_sixth_dof(
    module: str, exp_base: str, exp_ee: str
) -> None:
    """Trossen ViperX/WidowX put the 6th arm DOF (wrist-rotate) in
    ``gripper_link``. Auto-detect must keep it (6-DOF), not trim it as a gripper
    (which would give a 5-DOF chain that fails dispatch) -- the never-below-6-DOF
    guard (#470 follow-up)."""
    rd = pytest.importorskip(f"robot_descriptions.{module}")
    base, ee, _notes = suggest_base_ee_mjcf(rd.MJCF_PATH)
    assert (base, ee) == (exp_base, exp_ee)
    arm = Manipulator.from_mjcf(rd.MJCF_PATH)
    assert len(arm.kinbody.joints) == 6


@pytest.mark.parametrize(
    ("module", "exp_base", "exp_ee", "dof"),
    _REAL_MJCF_ARMS,
    ids=[a[0] for a in _REAL_MJCF_ARMS],
)
def test_strip_mjcf_to_fixture_preserves_kinematics(
    module: str, exp_base: str, exp_ee: str, dof: int, tmp_path: Path
) -> None:
    """The vendored kinematics-only MJCF is self-contained (no mesh/asset refs)
    and FK-identical to the source (#470)."""
    src = importlib.import_module(f"robot_descriptions.{module}").MJCF_PATH
    dest = tmp_path / "vendored.xml"
    # strip vendors the WHOLE model tree (so base/ee selection still works on the
    # fixture), so its joint count is >= the base->ee chain's dof (arx_l5 also has
    # a gripper joint).
    n_bodies, n_joints = strip_mjcf_to_fixture(Path(src), dest)
    assert n_bodies >= dof
    assert n_joints >= dof

    text = dest.read_text()
    assert "<mesh" not in text  # no asset tree needed
    assert "<asset" not in text

    base, ee, _ = suggest_base_ee_mjcf(dest)
    assert (base, ee) == (exp_base, exp_ee)
    kb_src = load_mjcf_kinbody_normalized(src, exp_base, exp_ee)
    kb_vendored = load_mjcf_kinbody_normalized(dest, base, ee)
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(30):
        q = rng.uniform(-1.5, 1.5, size=dof)
        worst = max(
            worst,
            float(
                np.abs(
                    poe_forward_kinematics(kb_src, q) - poe_forward_kinematics(kb_vendored, q)
                ).max()
            ),
        )
    assert worst < 1e-11, f"{module}: vendored MJCF FK drifted by {worst:.2e}"


@pytest.mark.parametrize(
    ("module", "exp_base", "exp_ee", "dof"),
    _REAL_MJCF_ARMS,
    ids=[a[0] for a in _REAL_MJCF_ARMS],
)
def test_from_mjcf_real_arm_fk_matches_mujoco(
    module: str, exp_base: str, exp_ee: str, dof: int
) -> None:
    path = importlib.import_module(f"robot_descriptions.{module}").MJCF_PATH

    # Auto-detect picks the documented base/ee (gripper trimmed to the flange).
    base, ee, _notes = suggest_base_ee_mjcf(path)
    assert (base, ee) == (exp_base, exp_ee)

    arm = Manipulator.from_mjcf(path)  # auto-detect
    assert len(arm.kinbody.joints) == dof

    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    jids = _chain_jnt_ids(model, base, ee)
    assert len(jids) == dof
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee)

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        q = rng.uniform(-1.0, 1.0, size=dof)
        data.qpos[:] = 0.0
        for qi, jid in zip(q, jids, strict=True):
            data.qpos[int(model.jnt_qposadr[jid])] = qi
        mujoco.mj_forward(model, data)
        r_base = data.xmat[base_id].reshape(3, 3)
        p_base = data.xpos[base_id]
        ref = np.eye(4)
        ref[:3, :3] = r_base.T @ data.xmat[ee_id].reshape(3, 3)
        ref[:3, 3] = r_base.T @ (data.xpos[ee_id] - p_base)
        worst = max(worst, float(np.abs(arm.fk(q) - ref).max()))
    assert worst < 1e-11, f"{module}: from_mjcf FK off from mujoco by {worst:.2e}"
