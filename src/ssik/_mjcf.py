"""MJCF → KinBody adapter (#343, sub-task of #83).

Loads a MuJoCo MJCF via the optional ``mujoco`` package and builds a
**POE-normalized** :class:`ssik._kinbody.KinBody` for the chain between two
bodies. ``mujoco`` compiles the MJCF -- resolving ``<default>`` classes,
``<compiler>`` angle/coordinate settings, ``<include>``, keyframes -- so we
never hand-parse the XML. Its ``mj_forward`` pass at the reference configuration
(``qpos0``) exposes world-frame joint axes/anchors and body poses, which *is*
the POE form, so the normalized chain reads straight off the compiled model.

``mujoco`` is an optional dependency (the ``mjcf`` extra), mirroring ``urchin``
for URDF; importing this module is fine without it, only the loader needs it.

The produced KinBody is identical in form to :func:`ssik._urdf.load_urdf_kinbody_normalized`:
solvers and the dispatcher are format-agnostic.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import JointType, KinBody, build_poe_kinbody
from ssik._urdf import _GRIPPER_HINTS, _MIN_ARM_DOF

__all__ = [
    "load_mjcf_kinbody_normalized",
    "strip_mjcf_to_fixture",
    "suggest_base_ee_mjcf",
]


def _import_mujoco() -> object:
    try:
        import mujoco
    except ImportError as err:
        raise ImportError(
            "MJCF loading requires the optional 'mjcf' extra: "
            "`pip install ssik[mjcf]` (or `uv add mujoco`)."
        ) from err
    return mujoco


def suggest_base_ee_mjcf(mjcf_path: str | Path) -> tuple[str, str, list[str]]:
    """Suggest ``(base_body, ee_body)`` for the longest actuated chain in an MJCF,
    plus human-readable notes about alternatives.

    ``base_body`` is the parent body of the first actuated (hinge/slide) joint --
    leading fixed bodies (e.g. ``world -> robot_base``) fold into the base.
    ``ee_body`` is the body carrying the last actuated joint (the kinematic
    flange); trailing gripper/tool bodies are reported as notes, not chosen. The
    MJCF analogue of :func:`ssik._urdf.suggest_base_ee`. For multi-limb robots
    (humanoids) only the single longest chain is returned -- pass an explicit
    ``base``/``ee`` to select a different limb.

    :returns: ``(base_body, ee_body, notes)``; ``notes`` empty for an
        unambiguous single-chain arm.
    :raises ValueError: if the MJCF has no hinge/slide joints.
    """
    mujoco = _import_mujoco()
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))  # type: ignore[attr-defined]
    hinge = mujoco.mjtJoint.mjJNT_HINGE  # type: ignore[attr-defined]
    slide = mujoco.mjtJoint.mjJNT_SLIDE  # type: ignore[attr-defined]

    def name(bid: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"  # type: ignore[attr-defined]

    def n_actuated(bid: int) -> int:
        j0 = int(model.body_jntadr[bid])
        return sum(
            1
            for j in range(j0, j0 + int(model.body_jntnum[bid]))
            if int(model.jnt_type[j]) in (hinge, slide)
        )

    children: dict[int, list[int]] = {}
    for b in range(1, model.nbody):  # skip world (body 0)
        children.setdefault(int(model.body_parentid[b]), []).append(b)

    # Longest chain by number of actuated joints, over every world-rooted branch.
    best: tuple[int, list[int]] | None = None

    def walk(bid: int, path: list[int], nact: int) -> None:
        nonlocal best
        na = nact + n_actuated(bid)
        if best is None or na > best[0]:
            best = (na, list(path))
        for c in children.get(bid, []):
            walk(c, [*path, c], na)

    for root in children.get(0, []):
        walk(root, [root], 0)

    if best is None or best[0] == 0:
        raise ValueError("suggest_base_ee_mjcf: no hinge/slide joints found in MJCF")

    chain = best[1]
    actuated = [b for b in chain if n_actuated(b) > 0]
    base_body = name(int(model.body_parentid[actuated[0]]))

    # Cumulative actuated-joint count along the chain (dof if ee = chain[i]).
    cum: list[int] = []
    running = 0
    for b in chain:
        running += n_actuated(b)
        cum.append(running)

    # Trim trailing gripper/tool bodies back to the kinematic flange, but never
    # below _MIN_ARM_DOF: a body named like a gripper can still carry a real arm
    # DOF (Trossen ViperX/WidowX put the 6th joint, wrist-rotate, in
    # ``gripper_link``), and since ssik only solves 6/7-DOF, trimming to <6 is
    # never the intended flange -- it just fails dispatch (#470 follow-up).
    ee_idx = chain.index(actuated[-1])
    first_idx = chain.index(actuated[0])
    while (
        ee_idx > first_idx
        and any(h in name(chain[ee_idx]).lower() for h in _GRIPPER_HINTS)
        and (cum[-1] < _MIN_ARM_DOF or cum[ee_idx - 1] >= _MIN_ARM_DOF)
    ):
        ee_idx -= 1
    ee_body = name(chain[ee_idx])

    notes: list[str] = []
    trailing = [
        name(b)
        for b in chain[ee_idx + 1 :]
        if not any(h in name(b).lower() for h in _GRIPPER_HINTS)
    ]
    if trailing:
        notes.append(f"frames past {ee_body!r}: {trailing} (pass ee= to use one)")
    trimmed_grippers = [name(b) for b in chain[ee_idx + 1 :] if name(b) not in trailing]
    if trimmed_grippers:
        notes.append(
            f"trimmed gripper/tool bodies past {ee_body!r}: {trimmed_grippers} "
            "(pass ee= to include one if it's actually an arm DOF)"
        )
    return base_body, ee_body, notes


def load_mjcf_kinbody_normalized(
    mjcf_path: str | Path,
    base_body: str,
    ee_body: str,
) -> KinBody:
    """Load an MJCF and build a POE-normalized :class:`KinBody` for the chain
    from ``base_body`` to ``ee_body``.

    Only single-DOF joints (``hinge`` → revolute, ``slide`` → prismatic) are
    supported; bodies with no joint are fused into the next active joint's
    transform (the MJCF analogue of URDF fixed-joint fusion). ``ball`` / ``free``
    joints raise. ``q=0`` corresponds to MuJoCo's reference ``qpos0``.

    :raises ValueError: if a body name is absent or ``ee_body`` is not a
        descendant of ``base_body``.
    :raises NotImplementedError: on ball/free joints.
    """
    mujoco = _import_mujoco()
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))  # type: ignore[attr-defined]
    data = mujoco.MjData(model)  # type: ignore[attr-defined]
    mujoco.mj_forward(model, data)  # type: ignore[attr-defined]  # qpos defaults to qpos0

    def _body_id(name: str) -> int:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)  # type: ignore[attr-defined]
        if bid < 0:
            raise ValueError(f"MJCF has no body named {name!r}")
        return int(bid)

    base_id = _body_id(base_body)
    ee_id = _body_id(ee_body)

    # Walk ee → base via the body tree; the base body is the reference frame.
    chain: list[int] = []
    b = ee_id
    while b != base_id:
        chain.append(b)
        parent = int(model.body_parentid[b])
        if parent == b:  # reached the world root without hitting base
            raise ValueError(f"body {ee_body!r} is not a descendant of {base_body!r}")
        b = parent
    chain.reverse()  # base-child … ee

    r_base = np.asarray(data.xmat[base_id], dtype=np.float64).reshape(3, 3)
    p_base = np.asarray(data.xpos[base_id], dtype=np.float64)

    hinge = mujoco.mjtJoint.mjJNT_HINGE  # type: ignore[attr-defined]
    slide = mujoco.mjtJoint.mjJNT_SLIDE  # type: ignore[attr-defined]

    records: list[
        tuple[str, JointType, NDArray[np.float64], NDArray[np.float64], tuple[float, float] | None]
    ] = []
    prev_pos = np.zeros(3, dtype=np.float64)  # base origin in base frame
    for body_id in chain:
        j0 = int(model.body_jntadr[body_id])
        njnt = int(model.body_jntnum[body_id])
        for jid in range(j0, j0 + njnt):
            jtype = int(model.jnt_type[jid])
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint{jid}"  # type: ignore[attr-defined]
            if jtype not in (hinge, slide):
                raise NotImplementedError(
                    f"joint {jname!r}: only hinge/slide (single-DOF) joints are "
                    "supported by the MJCF adapter (got ball/free)."
                )
            # World-frame axis/anchor at qpos0 expressed in the base frame = POE.
            axis_base = r_base.T @ np.asarray(data.xaxis[jid], dtype=np.float64)
            anchor_base = r_base.T @ (np.asarray(data.xanchor[jid], dtype=np.float64) - p_base)
            joint_type: JointType = "revolute" if jtype == hinge else "prismatic"
            if bool(model.jnt_limited[jid]):
                lo, hi = (float(x) for x in model.jnt_range[jid])
                limits: tuple[float, float] | None = (lo, hi)
            else:
                limits = None
            records.append((jname, joint_type, axis_base, anchor_base - prev_pos, limits))
            prev_pos = anchor_base

    if not records:
        raise ValueError(f"chain from {base_body!r} to {ee_body!r} has no joints")

    # Trailing offset + home orientation on the last joint's T_right.
    r_ee = np.asarray(data.xmat[ee_id], dtype=np.float64).reshape(3, 3)
    p_ee = np.asarray(data.xpos[ee_id], dtype=np.float64)
    final_t_right = np.eye(4, dtype=np.float64)
    final_t_right[:3, 3] = r_base.T @ (p_ee - p_base) - prev_pos
    rot4 = np.eye(4, dtype=np.float64)
    rot4[:3, :3] = r_base.T @ r_ee
    final_t_right = final_t_right @ rot4

    return build_poe_kinbody(records, final_t_right, base_body, ee_body)


def strip_mjcf_to_fixture(source: Path, dest: Path) -> tuple[int, int]:
    """Vendor a kinematics-only, self-contained MJCF from ``source`` to ``dest``.

    The MJCF analogue of :func:`ssik._urdf.strip_urdf_to_fixture`: ``mujoco``
    compiles the source (resolving ``<include>`` / ``<default>`` / assets), then
    we *reconstruct* a minimal nested-``<body>`` model from the compiled kinematic
    tree -- body ``pos``/``quat``, each hinge/slide joint's ``pos``/``axis``/
    ``range``, and a unit placeholder ``<inertial>`` -- with no ``<geom>`` /
    ``<asset>`` / mesh references. The result is self-contained (no mesh files
    needed on disk) and recompiles to the *identical* kinematics, so a vendored
    fixture never drags along an ``assets/`` tree.

    :returns: ``(n_bodies, n_joints)`` in the reconstructed model.
    """
    mujoco = _import_mujoco()
    model = mujoco.MjModel.from_xml_path(str(source))  # type: ignore[attr-defined]
    jtype_xml = {
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",  # type: ignore[attr-defined]
        int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",  # type: ignore[attr-defined]
        int(mujoco.mjtJoint.mjJNT_BALL): "ball",  # type: ignore[attr-defined]
        int(mujoco.mjtJoint.mjJNT_FREE): "free",  # type: ignore[attr-defined]
    }

    def bname(i: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"  # type: ignore[attr-defined]

    def jname(i: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint{i}"  # type: ignore[attr-defined]

    children: dict[int, list[int]] = {}
    for b in range(1, model.nbody):
        children.setdefault(int(model.body_parentid[b]), []).append(b)

    def _vec(v: NDArray[np.float64]) -> str:
        return " ".join(f"{float(x):.12g}" for x in v)

    model_name = Path(source).stem
    lines = [
        f"<mujoco model={quoteattr(model_name)}>",
        '  <compiler angle="radian"/>',
        "  <worldbody>",
    ]
    n_bodies = 0
    n_joints = 0

    def emit(bid: int, indent: int) -> None:
        nonlocal n_bodies, n_joints
        n_bodies += 1
        pad = "  " * indent
        lines.append(
            f"{pad}<body name={quoteattr(bname(bid))} "
            f'pos="{_vec(model.body_pos[bid])}" quat="{_vec(model.body_quat[bid])}">'
        )
        lines.append(f'{pad}  <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>')
        j0 = int(model.body_jntadr[bid])
        for j in range(j0, j0 + int(model.body_jntnum[bid])):
            n_joints += 1
            jt = jtype_xml.get(int(model.jnt_type[j]), "hinge")
            attrs = (
                f"name={quoteattr(jname(j))} type={quoteattr(jt)} "
                f'pos="{_vec(model.jnt_pos[j])}" axis="{_vec(model.jnt_axis[j])}"'
            )
            if bool(model.jnt_limited[j]):
                attrs += f' range="{_vec(model.jnt_range[j])}"'
            lines.append(f"{pad}  <joint {attrs}/>")
        for c in children.get(bid, []):
            emit(c, indent + 1)
        lines.append(f"{pad}</body>")

    for root in children.get(0, []):
        emit(root, 2)
    lines += ["  </worldbody>", "</mujoco>", ""]

    dest.write_text("\n".join(lines))
    return n_bodies, n_joints


def _mjcf_adapter() -> object:
    """Build the MJCF :class:`~ssik._formats.FormatAdapter` (registered on import)."""
    from ssik._formats import FormatAdapter

    def _load(path: Path, base: str, ee: str, _options: Mapping[str, Any]) -> KinBody:
        return load_mjcf_kinbody_normalized(path, base, ee)

    def _suggest(path: Path, _options: Mapping[str, Any]) -> tuple[str, str, list[str]]:
        return suggest_base_ee_mjcf(path)

    def _vendor(source: Path, dest: Path, _options: Mapping[str, Any]) -> tuple[int, int]:
        return strip_mjcf_to_fixture(source, dest)

    return FormatAdapter(
        kind="mjcf",
        label="MuJoCo MJCF",
        extensions=(".mjcf", ".xml"),
        fixture_suffix=".xml",
        scaffold_loader_module="ssik._mjcf",
        scaffold_loader_func="load_mjcf_kinbody_normalized",
        load=_load,
        suggest_base_ee=_suggest,
        vendor=_vendor,
        root_tags=("mujoco",),
    )


from ssik._formats import register as _register  # noqa: E402

_register(_mjcf_adapter())  # type: ignore[arg-type]
