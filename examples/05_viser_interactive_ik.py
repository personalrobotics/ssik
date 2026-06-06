"""Interactive analytical-IK demo: drag a marker, watch every IK branch.

This is the headline visual of ssik's "all analytical branches" story.
A 3D transform handle in your browser drives the target end-effector
pose; on every update ssik returns every analytical IK solution that
reaches that pose, and the demo renders them as live arms:

* The branch closest to the previous joint state (wrap-to-π) is the
  **solid arm** -- the teleop / demo-collection path
  ``solve(T, max_solutions=1, q_seed=q_current)`` would have picked.
* Every other branch is a **ghost arm** at the same instant: same
  target pose, different elbow / wrist / shoulder configuration.

Toggle through arms in the GUI -- including the ones EAIK refuses (any
non-Pieper 6R, any 7R). The badge shows what EAIK does on each, so the
wedge is visible side-by-side with what ssik returns.

Visuals come from ``robot_descriptions`` where it has a match (full
URDF meshes); arms without an upstream description (Puma 560, JACO 2,
Kassow, FANUC CRX, big_yam, OpenArm, Rizon 10) fall back to a colored
joint-spheres-plus-capsules rendering driven by ssik's own POE FK, so
every prebuilt is visible. A few arms can opportunistically load a
local URDF for full meshes via ``ArmSpec.local_urdf_paths`` (e.g. JACO
2 from a sibling ``ada_ros2/ada_description`` checkout) -- see #310.

Run::

    pip install 'ssik[demo]'
    python examples/05_viser_interactive_ik.py

Open the printed URL.
"""

from __future__ import annotations

import contextlib
import importlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

from ssik._urdf import load_urdf_kinbody_normalized

# ---------------------------------------------------------------------------
# Arm roster.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    label: str
    module_name: str  # ssik.prebuilt.<module_name>
    eaik_status: str
    expected_max_branches: int
    # URDF mesh source. ``rd_description`` triggers
    # ``robot_descriptions.loaders.yourdfpy.load_robot_description``;
    # ``None`` triggers the primitive fallback.
    rd_description: str | None = None
    # Optional local URDF (with sibling mesh files) for arms not packaged
    # in ``robot_descriptions``. ``~`` is expanded. First existing path
    # wins; if none exist the renderer falls back to primitives. Use this
    # for arms like JACO 2 whose meshes ship in personal-robotics-lab
    # repos (``ada_ros2/ada_description``) but not upstream.
    local_urdf_paths: tuple[str, ...] = ()
    # The link in the rendered URDF whose world transform we treat as the
    # arm's "end effector". Required for meshed arms; ignored otherwise.
    # Used to compute the fixed rigid offset between ssik's POE frame and
    # the rendered URDF's world frame -- different upstream packages
    # orient base_link differently (e.g. 180° about Z), which would
    # otherwise put the marker far from the visible EE.
    render_ee_link: str = ""
    # Subset of the URDF's actuated joints (in URDF order) that ssik's
    # q-vector drives. ``None`` means "all actuated joints", which only
    # works when the URDF's actuated count == ``module.DOF``. When the
    # URDF has extra joints (Panda's 2 grippers, etc.), name the IK ones.
    ik_joint_names: tuple[str, ...] | None = None
    # ssik fixture URDF, used to build the KinBody for FK chain-walking
    # (needed for primitive fallback AND for the marker-to-base offset).
    ssik_fixture: str = ""
    ssik_base_link: str = ""
    ssik_ee_link: str = ""


# Tier 1 — full meshes via robot_descriptions.
# Tier 2 — kinematic skeleton via primitives.
ARMS: list[ArmSpec] = [
    # Tier 1: meshed.
    ArmSpec(
        label="UR5 — three-parallel 6R (Pieper)",
        module_name="ur5_ik",
        eaik_status="supported (4 µs / FK 1.5e-15 / 2-8 sols)",
        expected_max_branches=8,
        rd_description="ur5_description",
        render_ee_link="ee_link",
        ssik_fixture="tests/fixtures/ur5.urdf",
        ssik_base_link="base_link",
        ssik_ee_link="ee_link",
    ),
    ArmSpec(
        label="KUKA iiwa14 — SRS 7R",
        module_name="iiwa14_ik",
        eaik_status='refuses ("no 7R DH path")',
        expected_max_branches=24,
        rd_description="iiwa14_description",
    ),
    ArmSpec(
        label="Unitree Z1 — three-parallel 6R (UR-class)",
        module_name="z1_ik",
        eaik_status="supported (4 µs / FK 1.5e-15 / 4-8 sols)",
        expected_max_branches=8,
        rd_description="z1_description",
        render_ee_link="link06",
        ssik_fixture="tests/fixtures/z1.urdf",
        ssik_base_link="link00",
        ssik_ee_link="link06",
    ),
    ArmSpec(
        label="I2RT YAM — non-Pieper 6R",
        module_name="yam_ik",
        eaik_status='refuses ("6R-Unknown Kinematic Class")',
        expected_max_branches=8,
        rd_description="yam_description",
        render_ee_link="link_6",
        ssik_fixture="tests/fixtures/yam.urdf",
        ssik_base_link="base_link",
        ssik_ee_link="link_6",
    ),
    ArmSpec(
        label="Franka Panda — anthropomorphic 7R",
        module_name="franka_panda_ik",
        eaik_status='refuses ("Currently, only 1-6R robots are sol")',
        expected_max_branches=32,
        rd_description="panda_description",
        render_ee_link="panda_link8",
        ik_joint_names=(
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ),
    ),
    ArmSpec(
        label="Franka Research 3 — anthropomorphic 7R",
        module_name="fr3_ik",
        eaik_status='refuses ("Currently, only 1-6R robots are sol")',
        expected_max_branches=32,
        rd_description="fr3_description",
        render_ee_link="fr3_link8",
        ik_joint_names=(
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ),
        ssik_fixture="tests/fixtures/fr3.urdf",
        ssik_base_link="fr3_link0",
        ssik_ee_link="fr3_link8",
    ),
    ArmSpec(
        label="UFactory xArm6 — non-Pieper 6R",
        module_name="xarm6_ik",
        eaik_status='refuses ("6R-Unknown Kinematic Class")',
        expected_max_branches=12,
        rd_description="xarm6_description",
        render_ee_link="link_eef",
        ssik_fixture="tests/fixtures/xarm6.urdf",
        ssik_base_link="link_base",
        ssik_ee_link="link_eef",
    ),
    ArmSpec(
        label="UFactory xArm7 — non-SRS 7R",
        module_name="xarm7_ik",
        eaik_status='refuses ("no 7R DH path")',
        expected_max_branches=32,
        rd_description="xarm7_description",
    ),
    ArmSpec(
        label="AgileX PiPER — non-Pieper 6R",
        module_name="piper_ik",
        eaik_status='refuses ("6R-Unknown Kinematic Class")',
        expected_max_branches=8,
        rd_description="piper_description",
        render_ee_link="link6",
        ik_joint_names=(
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ),
        ssik_fixture="tests/fixtures/piper.urdf",
        ssik_base_link="base_link",
        ssik_ee_link="link6",
    ),
    ArmSpec(
        label="Flexiv Rizon 4 — non-SRS 7R",
        module_name="rizon4_ik",
        eaik_status='refuses ("Currently, only 1-6R robots are sol")',
        expected_max_branches=32,
        rd_description="rizon4_description",
        render_ee_link="flange",
        ssik_fixture="tests/fixtures/rizon4.urdf",
        ssik_base_link="base_link",
        ssik_ee_link="flange",
    ),
    ArmSpec(
        label="Kinova Gen3 — approximate-SRS 7R",
        module_name="gen3_ik",
        eaik_status='refuses ("Currently, only 1-6R robots are sol")',
        expected_max_branches=32,
        rd_description="gen3_description",
        ssik_fixture="tests/fixtures/gen3.urdf",
        ssik_base_link="base_link",
        ssik_ee_link="end_effector_link",
    ),
    # Tier 2: primitive fallback (no upstream description).
    ArmSpec(
        label="KUKA Puma 560 — Pieper 6R (spherical wrist)",
        module_name="puma560_ik",
        eaik_status="supported (4 µs / FK 2.7e-14 / 8 sols)",
        expected_max_branches=8,
        ssik_fixture="tests/fixtures/puma560.urdf",
        ssik_base_link="base_link",
        ssik_ee_link="wrist_3_link",
    ),
    ArmSpec(
        label="Kinova JACO 2 — non-Pieper 6R",
        module_name="jaco2_ik",
        eaik_status='refuses ("6R-Unknown Kinematic Class")',
        expected_max_branches=12,
        # jaco2_ik is specs-based; no URDF in ssik. For meshed rendering,
        # opportunistically load the j2n6s200 URDF from a local checkout
        # of personal-robotics-lab's ada_ros2 (the IK chain matches at
        # 1.5e-7 -- rigid base-offset). Falls back to primitives if the
        # file isn't present.
        local_urdf_paths=(
            "~/code/robot-code/ada_ros2/ada_description/urdf/j2n6s200_clean.urdf",
        ),
        render_ee_link="j2n6s200_end_effector",
        ik_joint_names=tuple(f"j2n6s200_joint_{i}" for i in range(1, 7)),
        ssik_fixture="",
        ssik_base_link="",
        ssik_ee_link="",
    ),
    ArmSpec(
        label="Kassow KR810 — non-SRS 7R",
        module_name="kassow_kr810_ik",
        eaik_status='refuses ("Currently, only 1-6R robots are sol")',
        expected_max_branches=32,
        ssik_fixture="tests/fixtures/kassow_kr810.urdf",
        ssik_base_link="base",
        ssik_ee_link="end_effector",
    ),
    ArmSpec(
        label="FANUC CRX-10iA/L — non-Pieper 6R",
        module_name="fanuc_crx10ial_ik",
        eaik_status='refuses ("6R-Unknown Kinematic Class")',
        expected_max_branches=12,
        ssik_fixture="tests/fixtures/fanuc_crx10ial.urdf",
        ssik_base_link="base_link",
        ssik_ee_link="tool0",
    ),
    ArmSpec(
        label="I2RT big_yam — non-Pieper 6R",
        module_name="big_yam_ik",
        eaik_status='refuses ("6R-Unknown Kinematic Class")',
        expected_max_branches=8,
        ssik_fixture="tests/fixtures/big_yam.urdf",
        ssik_base_link="base",
        ssik_ee_link="gripper",
    ),
    ArmSpec(
        label="Flexiv Rizon 10 — non-SRS 7R",
        module_name="rizon10_ik",
        eaik_status='refuses ("Currently, only 1-6R robots are sol")',
        expected_max_branches=32,
        ssik_fixture="tests/fixtures/rizon10.urdf",
        ssik_base_link="base_link",
        ssik_ee_link="flange",
    ),
    ArmSpec(
        label="Enactic OpenArm v2.0 (left) — non-SRS 7R",
        module_name="openarm_left_ik",
        eaik_status='refuses ("Currently, only 1-6R robots are sol")',
        expected_max_branches=24,
        ssik_fixture="tests/fixtures/openarm_left.urdf",
        ssik_base_link="openarm_left_base_link",
        ssik_ee_link="openarm_left_ee_base_link",
    ),
    ArmSpec(
        label="Enactic OpenArm v2.0 (right) — non-SRS 7R",
        module_name="openarm_right_ik",
        eaik_status='refuses ("Currently, only 1-6R robots are sol")',
        expected_max_branches=24,
        ssik_fixture="tests/fixtures/openarm_right.urdf",
        ssik_base_link="openarm_right_base_link",
        ssik_ee_link="openarm_right_ee_base_link",
    ),
]


# ---------------------------------------------------------------------------
# Visuals.
# ---------------------------------------------------------------------------


# Two visual roles, two colours -- the active branch reads first, the
# ghosts read as supporting context. Tuned against viser's default
# studio-grey backdrop for legibility on screenshots / GIFs.
ACTIVE_COLOR_RGBA: tuple[float, float, float, float] = (0.86, 0.18, 0.18, 1.0)
# Ghosts are identical to the active: every visible branch is a valid IK
# solution, no privileged "preferred" coloring. Shadow casting is still
# disabled per-ghost so they don't darken each other or the active.
GHOST_COLOR_RGBA: tuple[float, float, float, float] = ACTIVE_COLOR_RGBA


def wrap_distance(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Wrap-to-π joint-space distance -- the metric the closest-branch
    teleop pattern uses."""
    diff = (q_a - q_b + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.linalg.norm(diff))


# ---------------------------------------------------------------------------
# Arm renderers.
#
# Two implementations, same shape: each takes a server + name root + RGBA;
# both expose ``set_q(q)`` and ``remove()``. The active arm is one
# renderer; each ghost branch is another.
# ---------------------------------------------------------------------------


class _Renderer(Protocol):
    """Visual renderer for one arm config (active or ghost).

    ``base_offset`` is the rigid transform from ssik's POE-normalized frame
    to the rendered URDF's world frame: ``T_render(q) = base_offset @ T_ssik(q)``.
    For the primitive renderer it's identity; for the meshed renderer it's
    measured at load time by probing both FKs at a reference q.
    """

    base_offset: np.ndarray

    def set_q(self, q: np.ndarray) -> None: ...
    def remove(self) -> None: ...


def _matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> Viser ``(w, x, y, z)`` quaternion."""
    trace = R.trace()
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=float)


def _align_z_to(direction: np.ndarray) -> np.ndarray:
    """Quaternion that rotates ``(0, 0, 1)`` to align with ``direction``."""
    d = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    d = d / n
    cos_t = float(np.clip(d[2], -1.0, 1.0))
    if cos_t > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if cos_t < -1.0 + 1e-12:
        # 180° about any perpendicular axis -- pick X.
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(np.array([0.0, 0.0, 1.0]), d)
    axis = axis / float(np.linalg.norm(axis))
    angle = float(np.arccos(cos_t))
    half = angle * 0.5
    s = float(np.sin(half))
    return np.array([float(np.cos(half)), axis[0] * s, axis[1] * s, axis[2] * s])


class MeshArmRenderer:
    """Real-mesh URDF arm via ``robot_descriptions`` + ``ViserUrdf``.

    Also stores the rigid offset between ssik's POE frame and the rendered
    URDF's world frame so the marker visually aligns with the rendered EE.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        urdf,  # pre-loaded yourdfpy.URDF (cached at module level)
        root_name: str,
        color_rgba: tuple[float, float, float, float],
        ik_joint_names: tuple[str, ...] | None,
        render_ee_link: str,
        ssik_fk: callable[[np.ndarray], np.ndarray],
        cast_shadow: bool = True,
    ) -> None:
        self._urdf = urdf
        self._render_ee_link = render_ee_link
        self.viz = ViserUrdf(
            server,
            urdf_or_path=urdf,
            root_node_name=root_name,
            mesh_color_override=color_rgba,
        )
        # Ghosts (cast_shadow=False) must not darken the active arm. The
        # public API doesn't expose this, but the per-link handles live
        # on ``viz._meshes`` and each one carries a settable attribute.
        if not cast_shadow:
            for handle in getattr(self.viz, "_meshes", []):
                with contextlib.suppress(Exception):
                    handle.cast_shadow = False
        self._urdf_joint_names = list(self.viz.get_actuated_joint_names())
        self._dof = len(self._urdf_joint_names)
        if ik_joint_names is None:
            self._ik_to_urdf = list(range(self._dof))
            self._ik_dof = self._dof
        else:
            self._ik_to_urdf = [
                self._urdf_joint_names.index(n) for n in ik_joint_names
            ]
            self._ik_dof = len(ik_joint_names)

        # Compute base_offset: a rigid transform such that
        #   T_render(q) = base_offset @ T_ssik(q)
        # holds at any q. Probe at a non-singular reference q.
        q_probe = np.full(self._ik_dof, 0.4)
        if self._ik_dof >= 4:
            q_probe[1] = 0.8
            q_probe[2] = -0.5
        T_ssik = ssik_fk(q_probe)
        T_render = self._render_fk(q_probe)
        self.base_offset = T_render @ _invert(T_ssik)

    def _render_fk(self, q: np.ndarray) -> np.ndarray:
        cfg = {n: 0.0 for n in self._urdf_joint_names}
        for i, urdf_idx in enumerate(self._ik_to_urdf):
            cfg[self._urdf_joint_names[urdf_idx]] = float(q[i])
        self._urdf.update_cfg(cfg)
        return np.asarray(self._urdf.get_transform(self._render_ee_link))

    def set_q(self, q: np.ndarray) -> None:
        cfg = np.zeros(self._dof, dtype=float)
        for i, urdf_idx in enumerate(self._ik_to_urdf):
            cfg[urdf_idx] = q[i]
        self.viz.update_cfg(cfg)

    def set_visible(self, visible: bool) -> None:
        self.viz.show_visual = visible

    def remove(self) -> None:
        self.viz.remove()


def _invert(T: np.ndarray) -> np.ndarray:
    """Rigid-transform inverse: ``[R | t] -> [R^T | -R^T t]``."""
    Ti = np.eye(4, dtype=float)
    R = T[:3, :3]
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


class PrimitiveArmRenderer:
    """Fallback renderer: joint-spheres + capsule bones, driven directly
    off ssik's POE-normalized KinBody.

    Since the primitives are drawn in ssik's POE frame, no base offset
    is needed -- it's identity.
    """

    JOINT_RADIUS = 0.045
    BONE_RADIUS = 0.025

    def __init__(
        self,
        server: viser.ViserServer,
        kb,  # ssik KinBody
        root_name: str,
        color_rgba: tuple[float, float, float, float],
        cast_shadow: bool = True,
    ) -> None:
        self.server = server
        self.kb = kb
        self.root = root_name
        self.base_offset = np.eye(4, dtype=float)
        self.color_rgb = (
            int(color_rgba[0] * 255),
            int(color_rgba[1] * 255),
            int(color_rgba[2] * 255),
        )
        self.opacity = float(color_rgba[3])
        self._n = len(kb.joints)

        # Build template meshes once. Bones are unit cylinders along z;
        # we stretch + rotate at update time.
        self._sphere_tmpl = trimesh.creation.icosphere(
            radius=self.JOINT_RADIUS, subdivisions=2
        )
        self._bone_tmpl_unit = trimesh.creation.cylinder(
            radius=self.BONE_RADIUS, height=1.0
        )

        # Pre-create N+1 joint spheres + N bones. Position/orientation
        # filled in by set_q(). Ghosts pass ``cast_shadow=False`` so
        # their shadows don't darken the active arm.
        self.joint_handles: list[viser.SceneNodeHandle] = []
        for i in range(self._n + 1):
            h = server.scene.add_icosphere(
                f"{root_name}/j{i}",
                radius=self.JOINT_RADIUS,
                color=self.color_rgb,
                opacity=self.opacity,
                cast_shadow=cast_shadow,
            )
            self.joint_handles.append(h)
        self.bone_handles: list[viser.SceneNodeHandle] = []
        for i in range(self._n):
            h = server.scene.add_mesh_simple(
                f"{root_name}/b{i}",
                vertices=self._bone_tmpl_unit.vertices,
                faces=self._bone_tmpl_unit.faces,
                color=self.color_rgb,
                opacity=self.opacity,
                cast_shadow=cast_shadow,
            )
            self.bone_handles.append(h)

    def set_q(self, q: np.ndarray) -> None:
        # Walk the chain, recording the rotation center of each joint
        # in the base frame.
        T = np.eye(4)
        centers = [T[:3, 3].copy()]
        for i, j in enumerate(self.kb.joints):
            T = T @ np.asarray(j.T_left)
            # Joint rotation about local axis by q[i].
            axis = np.asarray(j.axis)
            ang = float(q[i])
            c, s = float(np.cos(ang)), float(np.sin(ang))
            K = np.array(
                [
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0],
                ]
            )
            R = np.eye(3) + s * K + (1 - c) * (K @ K)
            R4 = np.eye(4)
            R4[:3, :3] = R
            T = T @ R4 @ np.asarray(j.T_right)
            centers.append(T[:3, 3].copy())

        # Spheres at every joint center (and EE).
        for h, p in zip(self.joint_handles, centers, strict=False):
            h.position = p

        # Bones between consecutive centers.
        for i, h in enumerate(self.bone_handles):
            a, b = centers[i], centers[i + 1]
            direction = b - a
            length = float(np.linalg.norm(direction))
            if length < 1e-9:
                h.visible = False
                continue
            h.visible = True
            h.position = ((a + b) * 0.5)
            h.scale = (1.0, 1.0, length)
            h.wxyz = _align_z_to(direction)

    def set_visible(self, visible: bool) -> None:
        for h in self.joint_handles + self.bone_handles:
            h.visible = visible

    def remove(self) -> None:
        for h in self.joint_handles + self.bone_handles:
            with contextlib.suppress(Exception):
                h.remove()


# ---------------------------------------------------------------------------
# Demo state.
# ---------------------------------------------------------------------------


def _load_ssik_kb(spec: ArmSpec, module):
    """KinBody for the primitive fallback path. Prefer the prebuilt's
    baked ``_KB`` (specs-only arms); else load the fixture URDF."""
    if hasattr(module, "_KB"):
        return module._KB
    if spec.ssik_fixture:
        return load_urdf_kinbody_normalized(
            spec.ssik_fixture, base_link=spec.ssik_base_link, ee_link=spec.ssik_ee_link
        )
    raise RuntimeError(
        f"{spec.module_name}: no _KB and no fixture URDF; cannot render"
    )


@dataclass
class ArmRuntime:
    spec: ArmSpec
    module: object
    active: _Renderer
    ghosts: list[_Renderer]
    dof: int
    q_current: np.ndarray
    last_sols_q: list[np.ndarray] = field(default_factory=list)

    def remove(self) -> None:
        self.active.remove()
        for g in self.ghosts:
            g.remove()


_URDF_CACHE: dict[str, object] = {}


def _get_urdf(description: str):
    """Cache loaded URDFs by description name. ``robot_descriptions``
    caches downloads but xacro parsing is fresh per call; caching the
    yourdfpy.URDF object avoids parsing N times when we instantiate one
    active + (N-1) ghost arms of the same robot."""
    cached = _URDF_CACHE.get(description)
    if cached is not None:
        return cached
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    cached = load_robot_description(description)
    _URDF_CACHE[description] = cached
    return cached


def _get_local_urdf(path: Path):
    """Load a local URDF (with sibling meshes). yourdfpy resolves relative
    ``filename`` references against the URDF's own directory, so a
    standard ROS-style ``../meshes/foo.dae`` Just Works."""
    key = f"local:{path}"
    cached = _URDF_CACHE.get(key)
    if cached is not None:
        return cached
    import yourdfpy

    cached = yourdfpy.URDF.load(str(path))
    _URDF_CACHE[key] = cached
    return cached


def _resolve_local_urdf(spec: ArmSpec) -> Path | None:
    """Return the first existing path from ``spec.local_urdf_paths`` (with
    ``~`` expansion), or ``None`` if no candidate file exists."""
    for raw in spec.local_urdf_paths:
        p = Path(raw).expanduser()
        if p.exists():
            return p
    return None


def preload_descriptions() -> None:
    """Warm the URDF cache for every meshed arm in the roster. Called
    from a background thread at startup so the first arm-switch isn't
    waiting on download + xacro for each one."""
    for spec in ARMS:
        local = _resolve_local_urdf(spec)
        if local is not None:
            try:
                _get_local_urdf(local)
            except Exception as e:
                print(f"  ! preload failed for {local}: {type(e).__name__}: {e}")
            continue
        if spec.rd_description is None:
            continue
        try:
            _get_urdf(spec.rd_description)
        except Exception as e:
            print(
                f"  ! preload failed for {spec.rd_description}: "
                f"{type(e).__name__}: {e}"
            )


def _mesh_renderer_has_geometry(renderer: MeshArmRenderer) -> bool:
    """Return True iff the underlying yourdfpy URDF has at least one
    resolvable visual mesh. Arms whose URDF references unresolvable
    ``package://...`` paths (e.g. yam_description's gripper assets)
    load as ViserUrdf with zero scene geometry -- the user sees nothing.
    Detect that and fall back to primitives instead of shipping an
    empty arm slot."""
    geom = getattr(renderer._urdf, "scene", None)
    if geom is None:
        return False
    return len(geom.geometry) > 0


def load_arm_runtime(server: viser.ViserServer, spec: ArmSpec) -> ArmRuntime:
    module = importlib.import_module(f"ssik.prebuilt.{spec.module_name}")
    dof = module.DOF
    n_ghosts = max(spec.expected_max_branches - 1, 0)

    # Two mesh sources: ``rd_description`` (upstream-packaged) or a
    # ``local_urdf_paths`` candidate that exists on disk (e.g. JACO 2's
    # meshes from ada_ros2/ada_description). Pick the first one
    # available; fall back to primitives if neither resolves.
    local_urdf = _resolve_local_urdf(spec)
    use_mesh = spec.rd_description is not None or local_urdf is not None
    # The rendered EE link MUST match the link the ssik artifact was built
    # for -- otherwise the rigid base-offset model can't account for the
    # tip-frame difference and the marker drifts as joints rotate. Default
    # to the manifest's ``ee_link`` (one source of truth) and only honor
    # ``spec.render_ee_link`` as an explicit override.
    from ssik.prebuilt._manifest import load_manifest

    manifest_ee = load_manifest()[spec.module_name].ee_link
    render_ee_link = spec.render_ee_link or manifest_ee

    def _build(root: str, rgba, cast_shadow: bool) -> _Renderer:
        if use_mesh:
            try:
                urdf = (
                    _get_local_urdf(local_urdf)
                    if local_urdf is not None
                    else _get_urdf(spec.rd_description)  # type: ignore[arg-type]
                )
                mesh = MeshArmRenderer(
                    server,
                    urdf,
                    root,
                    rgba,
                    spec.ik_joint_names,
                    render_ee_link,
                    module.fk,
                    cast_shadow=cast_shadow,
                )
                if _mesh_renderer_has_geometry(mesh):
                    return mesh
                mesh.remove()
                print(
                    f"  ! {spec.label}: rendered URDF has 0 visual meshes "
                    "(unresolvable package:// paths) -- falling back to primitives"
                )
            except Exception as e:
                print(
                    f"  ! mesh load failed for {spec.label}: "
                    f"{type(e).__name__}: {e}  -- falling back to primitives"
                )
        kb = _load_ssik_kb(spec, module)
        return PrimitiveArmRenderer(
            server, kb, root, rgba, cast_shadow=cast_shadow
        )

    active = _build("/arm/active", ACTIVE_COLOR_RGBA, cast_shadow=True)
    ghosts: list[_Renderer] = [
        _build(f"/arm/ghost_{i:02d}", GHOST_COLOR_RGBA, cast_shadow=False)
        for i in range(n_ghosts)
    ]

    # Seed q at a mid-workspace non-singular config.
    q0 = np.full(dof, 0.4)
    if dof >= 4:
        q0[1] = 0.8
        q0[2] = -0.5
    active.set_q(q0)
    for g in ghosts:
        g.set_q(q0)

    return ArmRuntime(
        spec=spec,
        module=module,
        active=active,
        ghosts=ghosts,
        dof=dof,
        q_current=q0,
    )


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main(host: str = "0.0.0.0", port: int = 8080) -> None:
    server = viser.ViserServer(host=host, port=port)

    # Top-level (no folder collapse) so the live stats / dispatch badges
    # are visible without the user hunting through panels. Controls go
    # inside folders below.
    arm_dropdown = server.gui.add_dropdown(
        "Robot",
        options=[a.label for a in ARMS],
        initial_value=ARMS[0].label,
    )
    solver_badge = server.gui.add_markdown("**ssik**: (loading…)")
    eaik_badge = server.gui.add_markdown("**EAIK**: (loading…)")
    stats_md = server.gui.add_markdown("**Stats**: waiting for first solve…")
    # ``max_ghosts_slider`` is hot-rewired in ``select_arm`` to match the
    # incoming arm's ``expected_max_branches``; the initial bounds are a
    # placeholder for the first arm in the roster.
    max_ghosts_slider = server.gui.add_slider(
        "Ghost branches shown",
        min=0,
        max=max(ARMS[0].expected_max_branches - 1, 0),
        step=1,
        initial_value=max(ARMS[0].expected_max_branches - 1, 0),
    )
    show_ghosts_chk = server.gui.add_checkbox("Show ghost branches", True)
    cycle_btn = server.gui.add_button("Cycle preferred branch")
    reset_btn = server.gui.add_button("Reset marker → current EE")

    state: dict[str, object] = {"arm": None}

    marker = server.scene.add_transform_controls(
        "/ee_target",
        scale=0.18,
        depth_test=False,
        disable_sliders=True,
    )

    def _move_marker(T: np.ndarray) -> None:
        marker.position = T[:3, 3]
        marker.wxyz = _matrix_to_wxyz(T[:3, :3])

    def _marker_T() -> np.ndarray:
        w, x, y, z = marker.wxyz
        T = np.eye(4, dtype=float)
        T[:3, 3] = marker.position
        T[:3, :3] = np.array(
            [
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
            ]
        )
        return T

    def _solve_and_render() -> None:
        runtime = state["arm"]
        if runtime is None:
            return
        rt: ArmRuntime = runtime  # type: ignore[assignment]

        # Marker is in the rendered URDF's world frame; ssik solves in its
        # POE frame. ``base_offset`` is the constant rigid transform
        # between the two (computed at load time by the renderer). Apply
        # the inverse before solving so a marker drag corresponds to a
        # 1:1 EE motion in the visible scene.
        T_marker = _marker_T()
        T_solve = _invert(rt.active.base_offset) @ T_marker

        t0 = time.perf_counter()
        sols = rt.module.solve(  # type: ignore[union-attr]
            T_solve,
            max_solutions=rt.spec.expected_max_branches,
            respect_limits=False,
            q_seed=rt.q_current,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not sols:
            stats_md.content = (
                "**Branches**: 0 (target out of reach)\n\n"
                f"**Solve**: {elapsed_ms:.2f} ms"
            )
            return

        q_list = [np.asarray(s.q, dtype=float) for s in sols]
        rt.last_sols_q = q_list

        # Always closest-track: pick the branch nearest to the previous
        # active q. Cycle button advances ``q_current`` to the next branch
        # before this re-solves, so closest-tracking lands on the cycled-to
        # branch by construction -- no fixed-slot indexing (which would
        # alias to different physical branches across solves, since the
        # solver doesn't promise a stable enumeration order).
        dists = [wrap_distance(q, rt.q_current) for q in q_list]
        active_idx = int(np.argmin(dists))

        q_active = q_list[active_idx]
        rt.active.set_q(q_active)
        rt.q_current = q_active

        # Ghost-arm rendering: respect both the show-ghosts toggle and the
        # user-selected cap. Ghosts beyond the cap are HIDDEN, not parked
        # at q_active -- parking stacks N transparent meshes on the active
        # arm and washes its color out.
        show = show_ghosts_chk.value
        max_ghosts = int(max_ghosts_slider.value) if show else 0
        ghost_idx = 0
        for i, q in enumerate(q_list):
            if i == active_idx:
                continue
            if ghost_idx >= len(rt.ghosts):
                break
            visible = ghost_idx < max_ghosts
            rt.ghosts[ghost_idx].set_visible(visible)
            if visible:
                rt.ghosts[ghost_idx].set_q(q)
            ghost_idx += 1
        for i in range(ghost_idx, len(rt.ghosts)):
            rt.ghosts[i].set_visible(False)

        fks = [float(s.fk_residual) for s in sols]
        stats_md.content = (
            f"**Branches**: {len(sols)} (active = #{active_idx})\n\n"
            f"**FK closure**: min {min(fks):.2e}, max {max(fks):.2e}\n\n"
            f"**Solve**: {elapsed_ms:.2f} ms"
        )

    def select_arm(label: str) -> None:
        if state["arm"] is not None:
            state["arm"].remove()  # type: ignore[attr-defined]
        spec = next(a for a in ARMS if a.label == label)
        runtime = load_arm_runtime(server, spec)
        state["arm"] = runtime
        solver_name = getattr(runtime.module, "SOLVER_NAME", "?")
        has_mesh = spec.rd_description is not None or _resolve_local_urdf(spec) is not None
        viz_kind = "URDF meshes" if has_mesh else "kinematic primitives"
        solver_badge.content = (
            f"**ssik**: `{solver_name}`  ·  {runtime.dof}-DOF  ·  "
            f"`from ssik.prebuilt import {spec.module_name}`\n\n"
            f"**viz**: {viz_kind}"
        )
        eaik_badge.content = f"**EAIK**: {spec.eaik_status}"
        # Rebind the slider bounds to this arm's branch budget. Default
        # to "all ghosts on" -- the user can dial it down to remove
        # visual clutter.
        max_n = max(spec.expected_max_branches - 1, 0)
        max_ghosts_slider.max = max_n
        max_ghosts_slider.value = max_n
        # Initial marker: place at the *rendered* EE position so the
        # handle is co-located with the visible end-effector.
        T_ssik = runtime.module.fk(runtime.q_current)  # type: ignore[union-attr]
        T_render = runtime.active.base_offset @ T_ssik
        _move_marker(T_render)
        _solve_and_render()

    @marker.on_update
    def _(_):
        _solve_and_render()

    @arm_dropdown.on_update
    def _(_):
        select_arm(arm_dropdown.value)

    @cycle_btn.on_click
    def _(_):
        runtime = state["arm"]
        if runtime is None or not runtime.last_sols_q:  # type: ignore[union-attr]
            return
        rt: ArmRuntime = runtime  # type: ignore[assignment]
        # Find the current active in the last solve, advance to the next
        # branch by q-identity (not by slot index). Update q_current so
        # the upcoming closest-track solve locks onto the cycled-to
        # branch even if the solver reorders its enumeration.
        qs = rt.last_sols_q
        cur_idx = int(np.argmin([wrap_distance(q, rt.q_current) for q in qs]))
        next_idx = (cur_idx + 1) % len(qs)
        rt.q_current = qs[next_idx]
        _solve_and_render()
        _solve_and_render()

    @reset_btn.on_click
    def _(_):
        runtime = state["arm"]
        if runtime is None:
            return
        rt: ArmRuntime = runtime  # type: ignore[assignment]
        T_ssik = rt.module.fk(rt.q_current)  # type: ignore[union-attr]
        T_render = rt.active.base_offset @ T_ssik
        _move_marker(T_render)
        _solve_and_render()

    @show_ghosts_chk.on_update
    def _(_):
        _solve_and_render()

    @max_ghosts_slider.on_update
    def _(_):
        _solve_and_render()

    # Warm the URDF cache for the rest of the meshed roster in the
    # background so the first switch through each arm doesn't pay the
    # download + xacro cost interactively. First-ever launch of an arm
    # lazy ``git clone``s its upstream description repo into
    # ``~/.cache/robot_descriptions/`` -- ~100s of MB total across the
    # full roster. Subsequent launches reuse the cache and are instant.
    import threading

    threading.Thread(target=preload_descriptions, daemon=True).start()

    select_arm(ARMS[0].label)
    print(f"\n  ssik interactive-IK demo:  http://localhost:{port}", flush=True)
    print(
        "  (first launch: upstream URDFs lazy-fetch to "
        "~/.cache/robot_descriptions/ in the background)\n",
        flush=True,
    )

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8080, type=int)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
