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

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import JointType, KinBody, build_poe_kinbody

__all__ = ["load_mjcf_kinbody_normalized"]


def _import_mujoco() -> object:
    try:
        import mujoco
    except ImportError as err:
        raise ImportError(
            "MJCF loading requires the optional 'mjcf' extra: "
            "`pip install ssik[mjcf]` (or `uv add mujoco`)."
        ) from err
    return mujoco


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
