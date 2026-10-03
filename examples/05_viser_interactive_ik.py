"""Interactive analytical-IK demo: drag a marker, watch every IK branch.

This is the headline visual of ssik's "all analytical branches" story.
A 3D transform handle in your browser drives the target end-effector
pose. An :class:`ssik.Tracker` follows it, and the demo renders every
analytical IK solution at that pose as a live arm:

* The **solid arm** is the tracker's configuration: the branch nearest
  the previous one, continued through singularities and kept inside the
  joint limits -- what a teleop loop would command.
* Every other branch is a **ghost arm** at the same instant: same
  target pose, different elbow / wrist / shoulder configuration
  (``Tracker.solutions()``). "Cycle preferred branch" switches the solid
  arm to the next one (``Tracker.next_branch()``).

Toggle through arms in the GUI -- including the ones EAIK refuses (any
non-Pieper 6R, any 7R). The badge shows what EAIK does on each (measured,
from ssik's prebuilt manifest), so the wedge is visible side-by-side with
what ssik returns.

Visuals come from ``robot_descriptions`` where it has a match (full
URDF meshes); arms without an upstream description (Puma 560, JACO 2,
Kassow, FANUC CRX, big_yam, OpenArm, Rizon 10), and any whose
description fails to load, fall back to a colored joint-spheres-plus-
capsules rendering driven by ssik's own FK, so every arm is visible.
Descriptions that are xacro need ``pip install 'ssik[xacro]'``; without
it those arms (FR3, xArm, Rizon 4, Gen3) render as primitives. A local
URDF with meshes can be supplied through ``ArmSpec.local_urdf_paths``
(see #310). ``--no-meshes`` forces primitives everywhere (offline, or
headless).

The marker lives in the rendered scene's frame; the rigid offset between
that frame and the arm's base frame is a calibration
(``ssik.teleop.apply_calibration``), the same step a VR or SpaceMouse
loop takes (``examples/07_teleop.py``).

Run::

    pip install 'ssik[demo]'
    python examples/05_viser_interactive_ik.py

Open the printed URL.
"""

from __future__ import annotations

import contextlib
import importlib
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

import ssik
from ssik.teleop import apply_calibration, invert

# ---------------------------------------------------------------------------
# Arm roster.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    label: str
    module_name: str  # a name ``ssik.Manipulator.from_prebuilt`` resolves
    # URDF mesh source. ``rd_description`` triggers
    # ``robot_descriptions.loaders.yourdfpy.load_robot_description``;
    # ``None`` triggers the primitive fallback.
    rd_description: str | None = None
    # Optional local URDF (with sibling mesh files) for arms not packaged
    # in ``robot_descriptions``. ``~`` is expanded. First existing path
    # wins; if none exist the renderer falls back to primitives.
    local_urdf_paths: tuple[str, ...] = ()
    # The link in the rendered URDF whose world transform we treat as the
    # arm's "end effector". Defaults to the artifact's ``EE_LINK``. Used
    # to compute the fixed rigid offset between ssik's base frame and the
    # rendered URDF's world frame -- different upstream packages orient
    # base_link differently (e.g. 180° about Z), which would otherwise put
    # the marker far from the visible EE.
    render_ee_link: str = ""
    # Subset of the URDF's actuated joints (in URDF order) that ssik's
    # q-vector drives. ``None`` means "all actuated joints", which only
    # works when the URDF's actuated count == the arm's DOF. When the
    # URDF has extra joints (Panda's 2 grippers, etc.), name the IK ones.
    ik_joint_names: tuple[str, ...] | None = None


# Tier 1 — full meshes via robot_descriptions.
# Tier 2 — kinematic skeleton via primitives.
ARMS: list[ArmSpec] = [
    # Tier 1: meshed.
    ArmSpec(
        label="UR5 — three-parallel 6R (Pieper)",
        module_name="ur5_ik",
        rd_description="ur5_description",
        render_ee_link="ee_link",
    ),
    ArmSpec(
        label="KUKA iiwa14 — SRS 7R",
        module_name="iiwa14_ik",
        rd_description="iiwa14_description",
    ),
    ArmSpec(
        label="Unitree Z1 — three-parallel 6R (UR-class)",
        module_name="z1_ik",
        rd_description="z1_description",
        render_ee_link="link06",
    ),
    ArmSpec(
        label="I2RT YAM — non-Pieper 6R",
        module_name="yam_ik",
        rd_description="yam_description",
        render_ee_link="link_6",
    ),
    ArmSpec(
        label="Franka Panda — anthropomorphic 7R",
        module_name="franka_panda_ik",
        rd_description="panda_description",
        render_ee_link="panda_link8",
        ik_joint_names=tuple(f"panda_joint{i}" for i in range(1, 8)),
    ),
    ArmSpec(
        label="Franka Research 3 — anthropomorphic 7R",
        module_name="fr3_ik",
        rd_description="fr3_description",
        render_ee_link="fr3_link8",
        ik_joint_names=tuple(f"fr3_joint{i}" for i in range(1, 8)),
    ),
    ArmSpec(
        label="UFactory xArm6 — non-Pieper 6R",
        module_name="xarm6_ik",
        rd_description="xarm6_description",
        render_ee_link="link_eef",
    ),
    ArmSpec(
        label="UFactory xArm7 — non-SRS 7R",
        module_name="xarm7_ik",
        rd_description="xarm7_description",
    ),
    ArmSpec(
        label="AgileX PiPER — non-Pieper 6R",
        module_name="piper_ik",
        rd_description="piper_description",
        render_ee_link="link6",
        ik_joint_names=tuple(f"joint{i}" for i in range(1, 7)),
    ),
    ArmSpec(
        label="Flexiv Rizon 4 — non-SRS 7R",
        module_name="rizon4_ik",
        rd_description="rizon4_description",
        render_ee_link="flange",
    ),
    ArmSpec(
        label="Kinova Gen3 — approximate-SRS 7R",
        module_name="gen3_ik",
        rd_description="gen3_description",
    ),
    # Tier 2: primitive fallback (no upstream description).
    ArmSpec(label="KUKA Puma 560 — Pieper 6R (spherical wrist)", module_name="puma560_ik"),
    ArmSpec(
        label="Kinova JACO 2 — non-Pieper 6R",
        module_name="jaco2_ik",
        # No upstream description. To render meshes, list a local copy of
        # personal-robotics-lab's ada_description j2n6s200 URDF here (the IK
        # chain matches it to 1.5e-7, a rigid base offset).
        render_ee_link="j2n6s200_end_effector",
        ik_joint_names=tuple(f"j2n6s200_joint_{i}" for i in range(1, 7)),
    ),
    ArmSpec(label="Kassow KR810 — non-SRS 7R", module_name="kassow_kr810_ik"),
    ArmSpec(label="FANUC CRX-10iA/L — non-Pieper 6R", module_name="fanuc_crx10ial_ik"),
    ArmSpec(label="I2RT big_yam — non-Pieper 6R", module_name="big_yam_ik"),
    ArmSpec(label="Flexiv Rizon 10 — non-SRS 7R", module_name="rizon10_ik"),
    ArmSpec(label="Enactic OpenArm v2.0 (left) — non-SRS 7R", module_name="openarm_left_ik"),
    ArmSpec(label="Enactic OpenArm v2.0 (right) — non-SRS 7R", module_name="openarm_right_ik"),
]

# Ghost-arm slots are created when an arm loads, one renderer each. This caps
# them on the arms whose solve() samples a self-motion into many points.
MAX_BRANCHES = 32


def _manifest_arms() -> dict[str, Any]:
    """The prebuilt manifest (``ssik/prebuilt/MANIFEST.toml``, shipped in the
    wheel): the measured EAIK comparison for each arm lives there."""
    from importlib.resources import files

    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib  # type: ignore[no-redef]

    text = (files("ssik.prebuilt") / "MANIFEST.toml").read_text(encoding="utf-8")
    arms: dict[str, Any] = tomllib.loads(text)["arms"]
    return arms


def eaik_status(module_name: str) -> str:
    """What EAIK does on this arm, from the measured manifest entry."""
    entry = _manifest_arms().get(module_name, {}).get("eaik")
    if not entry:
        return "not measured"
    if entry.get("supported"):
        return (
            f"supported ({entry['ms_mean'] * 1000:.0f} µs / FK {entry['max_fk']:.1e} / "
            f"{entry['sols_min']}-{entry['sols_max']} sols)"
        )
    return f'refuses ("{entry.get("refusal", "?")}")'


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
            self._ik_to_urdf = [self._urdf_joint_names.index(n) for n in ik_joint_names]
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
        self.base_offset = T_render @ invert(T_ssik)

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
        # ViserUrdf.remove() removes each link frame before the meshes under
        # it, so every mesh removal warns that its node is already gone
        # (~30 warnings per arm, thousands over a tour). The scene is
        # correct; only the warning is noise.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Attempted to remove already removed node")
            self.viz.remove()


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
        self._sphere_tmpl = trimesh.creation.icosphere(radius=self.JOINT_RADIUS, subdivisions=2)
        self._bone_tmpl_unit = trimesh.creation.cylinder(radius=self.BONE_RADIUS, height=1.0)

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
            h.position = (a + b) * 0.5
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


@dataclass
class ArmRuntime:
    spec: ArmSpec
    arm: ssik.Manipulator
    tracker: ssik.Tracker
    active: _Renderer
    ghosts: list[_Renderer]
    # Per-ghost-slot last-rendered q, for stable identity tracking across
    # solves. None means "this slot hasn't been bound to a branch yet".
    # See the render loop -- each frame we greedy-match each slot to the
    # closest q in the new solve, so the solver's enumeration order
    # changing across frames doesn't make ghost #5 flicker between two
    # different physical branches.
    ghost_qs: list[np.ndarray | None] = field(default_factory=list)

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


def preload_descriptions(specs: list[ArmSpec]) -> None:
    """Warm the URDF cache for the meshed arms in ``specs``. Called from a
    background thread at startup so the first arm-switch isn't waiting on
    download + xacro for each one."""
    for spec in specs:
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
            print(f"  ! preload failed for {spec.rd_description}: {type(e).__name__}: {e}")


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


def _start_q(arm: ssik.Manipulator) -> np.ndarray:
    """A mid-workspace, non-singular start inside every joint's limits."""
    q0 = np.full(arm.dof, 0.4)
    if arm.dof >= 4:
        q0[1] = 0.8
        q0[2] = -0.5
    for i, lim in enumerate(arm.joint_limits):
        if lim is not None:
            lo, hi = lim
            margin = 0.1 * (hi - lo)
            q0[i] = float(np.clip(q0[i], lo + margin, hi - margin))
    return q0


def _branch_slots(arm: ssik.Manipulator, q0: np.ndarray) -> int:
    """How many arms to create for this robot: its branch count at the start
    pose (one per geometric branch, limits ignored), at least eight (a
    Pieper 6R's count) and at most ``MAX_BRANCHES``."""
    sols = arm.solve(
        arm.fk(q0), respect_limits=False, enumerate_windings=False, max_solutions=MAX_BRANCHES
    )
    return min(max(len(sols), 8), MAX_BRANCHES)


def load_arm_runtime(server: viser.ViserServer, spec: ArmSpec, *, meshes: bool) -> ArmRuntime:
    arm = ssik.Manipulator.from_prebuilt(spec.module_name)
    info = next(a for a in ssik.list_arms() if a.name == spec.module_name)
    q0 = _start_q(arm)
    n_ghosts = _branch_slots(arm, q0) - 1

    # Two mesh sources: ``rd_description`` (upstream-packaged) or a
    # ``local_urdf_paths`` candidate that exists on disk. Pick the first one
    # available; fall back to primitives if neither resolves.
    local_urdf = _resolve_local_urdf(spec)
    use_mesh = meshes and (spec.rd_description is not None or local_urdf is not None)
    # The rendered EE link MUST match the link the ssik artifact was built
    # for -- otherwise the rigid base-offset model can't account for the
    # tip-frame difference and the marker drifts as joints rotate. Default
    # to the artifact's ``EE_LINK`` (one source of truth) and only honor
    # ``spec.render_ee_link`` as an explicit override.
    render_ee_link = spec.render_ee_link or importlib.import_module(info.import_path).EE_LINK

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
                    arm.fk,
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
        return PrimitiveArmRenderer(server, arm.kinbody, root, rgba, cast_shadow=cast_shadow)

    active = _build("/arm/active", ACTIVE_COLOR_RGBA, cast_shadow=True)
    ghosts: list[_Renderer] = [
        _build(f"/arm/ghost_{i:02d}", GHOST_COLOR_RGBA, cast_shadow=False) for i in range(n_ghosts)
    ]
    active.set_q(q0)
    for g in ghosts:
        g.set_q(q0)

    return ArmRuntime(
        spec=spec,
        arm=arm,
        # allow_jump: the solid arm is always the branch nearest the last one,
        # however far, as a demo wants. A teleop loop keeps the default and
        # holds instead (examples/07_teleop.py).
        tracker=arm.tracker(q0, allow_jump=True),
        active=active,
        ghosts=ghosts,
        ghost_qs=[None] * len(ghosts),
    )


_HELD_TEXT = {
    "unreachable": "target out of reach",
    "limits": "target outside the joint limits",
    "jump": "nearest branch too far",
}


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main(
    host: str = "127.0.0.1",
    port: int = 8080,
    meshes: bool = True,
    tour: bool = False,
    tour_delay_s: float = 4.0,
    tour_per_arm_s: float = 3.0,
    tour_settle_s: float = 2.5,
    tour_exit: bool = False,
    tour_max_ghosts: int = 7,
    tour_record_dir: str = "",
    tour_record_size: tuple[int, int] = (1280, 720),
    self_motion: bool = False,
    self_motion_arm: str = "Franka Panda — anthropomorphic 7R",
    self_motion_seconds: float = 8.0,
    self_motion_seed: int = 23,
    self_motion_ghosts: int = 0,
) -> None:
    server = viser.ViserServer(host=host, port=port)

    # Viser runs GUI callbacks on a thread pool, so a drag, a slider move and
    # an arm switch can arrive at once. Every read or write of the demo state
    # (the loaded arm, its tracker, the renderers) holds this lock.
    lock = threading.RLock()

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
    # ``max_ghosts_slider`` is hot-rewired in ``select_arm`` to the incoming
    # arm's ghost slots; the initial bounds are a placeholder.
    max_ghosts_slider = server.gui.add_slider(
        "Ghost branches shown", min=0, max=7, step=1, initial_value=7
    )
    show_ghosts_chk = server.gui.add_checkbox("Show ghost branches", True)
    cycle_btn = server.gui.add_button("Cycle preferred branch")
    reset_btn = server.gui.add_button("Reset marker → current EE")

    state: dict[str, ArmRuntime | None] = {"arm": None}

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
        # The client sends a float32 quaternion; normalise it so the target
        # is a rotation to round-off, as solve() requires.
        w, x, y, z = np.asarray(marker.wxyz, dtype=float) / np.linalg.norm(marker.wxyz)
        T = np.eye(4, dtype=float)
        T[:3, 3] = marker.position
        T[:3, :3] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        return T

    def _solve_and_render() -> bool:
        """Return True if the tracker followed the marker and we rendered it;
        False if it held (the marker pose is out of reach). Tour mode uses
        this signal to skip capturing dead frames where the arm has frozen
        at its last reachable pose."""
        with lock:
            rt = state["arm"]
            if rt is None:
                return False
            # All scene updates inside this call are wrapped in
            # ``server.atomic()`` so the client receives one batched apply per
            # frame. Without this, per-joint updates (ViserUrdf.update_cfg
            # emits N separate messages) interleave between arms and the user
            # sees brief intermediate poses where the EE is visibly off-marker
            # -- the "flicker that isn't an IK solution" symptom.
            with server.atomic():
                return _solve_and_render_inner(rt)

    def _solve_and_render_inner(rt: ArmRuntime) -> bool:
        # The marker is in the rendered URDF's world frame; ssik solves in the
        # arm's base frame. ``base_offset`` (render_T_base, measured by the
        # renderer at load time) is that calibration; its inverse maps the
        # marker into the base frame, so a marker drag is a 1:1 EE motion in
        # the visible scene.
        T_solve = apply_calibration(invert(rt.active.base_offset), _marker_T())

        # Tie the ghost solve to what we actually render: at most the visible
        # ghosts + the active. The slider caps the rendered count; asking for
        # more would burn CPU on branches we'd never paint.
        show = show_ghosts_chk.value
        max_ghosts = min(int(max_ghosts_slider.value), len(rt.ghosts)) if show else 0
        t0 = time.perf_counter()
        step = rt.tracker.update(T_solve)
        others: list[ssik.Solution] = []
        if step.status is not ssik.TrackStatus.HELD and max_ghosts > 0:
            # Every other branch at this pose, nearest first; the tracker's own
            # configuration is the active arm, not a ghost.
            others = [
                s
                for s in rt.tracker.solutions(max_solutions=max_ghosts + 1)
                if float(np.max(np.abs(s.q - step.q))) > 1e-6
            ][:max_ghosts]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if step.status is ssik.TrackStatus.HELD:
            reason = _HELD_TEXT.get(step.reason or "", step.reason)
            stats_md.content = f"**Branches**: 0 ({reason})\n\n**Solve**: {elapsed_ms:.2f} ms"
            return False

        rt.active.set_q(step.q)

        # Ghost-arm rendering: respect both the show-ghosts toggle and the
        # user-selected cap. Ghosts beyond the cap are HIDDEN (not parked
        # at the active q -- parking stacks N transparent meshes on the active
        # and washes its color out).
        #
        # Stable identity via greedy matching: each ghost slot remembers
        # its prior q; we assign it the closest remaining q in the new
        # solve. The solver doesn't promise a stable enumeration order
        # across solves, so a positional "ghosts[i] = q_list[i]" assignment
        # makes ghost #5 visibly flicker between two unrelated branches as
        # the solver shuffles. Greedy-match keeps each ghost on its
        # continuation branch as long as that branch survives in the new
        # solve. Distances are plain coordinate distances: every q here is
        # the in-limit representative nearest the active arm.
        q_list = [np.asarray(s.q, dtype=float) for s in others]
        available = list(range(len(q_list)))
        n_slots = len(q_list)
        # Slot ordering: bind first to slots that already have a prior
        # q (so they stick to their continuation); fall through to fresh
        # slots seeded against the active q.
        seeded_slots = [s for s in range(n_slots) if rt.ghost_qs[s] is not None]
        fresh_slots = [s for s in range(n_slots) if rt.ghost_qs[s] is None]
        for slot in seeded_slots + fresh_slots:
            prior = rt.ghost_qs[slot]
            ref_q = prior if prior is not None else step.q
            best_j = min(available, key=lambda j: float(np.max(np.abs(q_list[j] - ref_q))))
            available.remove(best_j)
            q_pick = q_list[best_j]
            rt.ghosts[slot].set_visible(True)
            rt.ghosts[slot].set_q(q_pick)
            rt.ghost_qs[slot] = q_pick
        for slot in range(n_slots, len(rt.ghosts)):
            rt.ghosts[slot].set_visible(False)
            rt.ghost_qs[slot] = None

        fks = [step.fk_residual] + [float(s.fk_residual) for s in others]
        stats_md.content = (
            f"**Branches**: {1 + len(others)} shown ({step.status.name.lower()})\n\n"
            f"**FK closure**: min {min(fks):.2e}, max {max(fks):.2e}\n\n"
            f"**Solve**: {elapsed_ms:.2f} ms"
        )
        return True

    def select_arm(label: str) -> None:
        with lock:
            current = state["arm"]
            if current is not None and current.spec.label == label:
                return  # already loaded (the dropdown and the tour both ask)
            if current is not None:
                current.remove()
                state["arm"] = None
            spec = next(a for a in ARMS if a.label == label)
            runtime = load_arm_runtime(server, spec, meshes=meshes)
            state["arm"] = runtime
            has_mesh = meshes and (
                spec.rd_description is not None or _resolve_local_urdf(spec) is not None
            )
            viz_kind = "URDF meshes" if has_mesh else "kinematic primitives"
            solver_badge.content = (
                f"**ssik**: `{runtime.arm.solver_name}`  ·  {runtime.arm.dof}-DOF  ·  "
                f'`ssik.Manipulator.from_prebuilt("{spec.module_name}")`\n\n'
                f"**viz**: {viz_kind}"
            )
            eaik_badge.content = f"**EAIK**: {eaik_status(spec.module_name)}"
            # Rebind the slider bounds to this arm's ghost slots. Default to
            # "all ghosts on" -- the user can dial it down to remove clutter.
            max_n = len(runtime.ghosts)
            max_ghosts_slider.max = max_n
            max_ghosts_slider.value = max_n
            # Initial marker: place at the *rendered* EE so the handle is
            # co-located with the visible end-effector.
            _move_marker(runtime.active.base_offset @ runtime.arm.fk(runtime.tracker.q))
            _solve_and_render()

    @marker.on_update
    def _(_):
        _solve_and_render()

    @arm_dropdown.on_update
    def _(_):
        select_arm(arm_dropdown.value)

    @cycle_btn.on_click
    def _(_):
        with lock:
            rt = state["arm"]
            if rt is None:
                return
            # Switch the tracker to the next branch at the current target
            # (by configuration, not by slot index); the render then follows
            # it like any other update.
            rt.tracker.next_branch()
            _solve_and_render()

    @reset_btn.on_click
    def _(_):
        with lock:
            rt = state["arm"]
            if rt is None:
                return
            _move_marker(rt.active.base_offset @ rt.arm.fk(rt.tracker.q))
            _solve_and_render()

    @show_ghosts_chk.on_update
    def _(_):
        _solve_and_render()

    @max_ghosts_slider.on_update
    def _(_):
        _solve_and_render()

    # Warm the URDF cache for the meshed roster in the background so the
    # first switch through each arm doesn't pay the download + xacro cost
    # interactively. First-ever launch of an arm lazy ``git clone``s its
    # upstream description repo into ``~/.cache/robot_descriptions/`` --
    # ~100s of MB total across the full roster. Subsequent launches reuse
    # the cache and are instant.
    #
    # Tour mode runs the preload synchronously up front instead so the
    # background thread isn't competing for CPU / disk / GIL while we're
    # trying to drive frames at 30fps. Self-motion mode shows one arm, so it
    # loads only that one.
    if meshes and tour:
        print("  tour mode: pre-warming URDF cache (synchronous)", flush=True)
        preload_descriptions([a for a in ARMS if a.label in _TOUR_ORDER])
    elif meshes and not self_motion:
        threading.Thread(target=preload_descriptions, args=(ARMS,), daemon=True).start()

    select_arm(self_motion_arm if self_motion and not tour else ARMS[0].label)
    print(f"\n  ssik interactive-IK demo:  http://{host}:{server.get_port()}", flush=True)
    if meshes:
        print(
            "  (first launch: upstream URDFs lazy-fetch to "
            "~/.cache/robot_descriptions/ in the background)\n",
            flush=True,
        )

    try:
        if tour:
            # Tour mode: drive arm dropdown + marker programmatically through
            # a narrative of arms. ``tour_record_dir`` opts into per-frame PNG
            # capture via the connected client's server-driven ``get_render``;
            # ffmpeg later composes them into the final MP4 / GIF. With no
            # record dir we just animate live.
            print(
                f"  tour mode: starting in {tour_delay_s:.1f}s, "
                f"{tour_per_arm_s:.1f}s motion per arm",
                flush=True,
            )
            record_dir = _await_recorder(server, tour_record_dir, tour_record_size, "tour")
            time.sleep(tour_delay_s)
            _run_tour(
                select_arm=select_arm,
                arm_dropdown=arm_dropdown,
                max_ghosts_slider=max_ghosts_slider,
                move_marker=_move_marker,
                solve_and_render=_solve_and_render,
                state=state,
                lock=lock,
                per_arm_s=tour_per_arm_s,
                max_ghosts=tour_max_ghosts,
                server=server,
                record_dir=record_dir,
                record_size=tour_record_size,
                settle_after_load_s=tour_settle_s,
            )
            print("  tour: complete", flush=True)
            if tour_exit:
                return

        if self_motion:
            # Self-motion mode: one redundant arm, one fixed target, sweeping
            # a single branch of the self-motion manifold. Shares the tour's
            # capture path, so the same record dir feeds the same encoder.
            record_dir = _await_recorder(server, tour_record_dir, tour_record_size, "self-motion")
            time.sleep(tour_delay_s)
            _run_self_motion(
                select_arm=select_arm,
                move_marker=_move_marker,
                state=state,
                lock=lock,
                arm_label=self_motion_arm,
                seconds=self_motion_seconds,
                n_ghosts=self_motion_ghosts,
                server=server,
                record_dir=record_dir,
                record_size=tour_record_size,
                settle_s=tour_settle_s,
                seed=self_motion_seed,
            )
            print("  self-motion: complete", flush=True)
            if tour_exit:
                return

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n  stopping", flush=True)
    finally:
        server.stop()


def _await_recorder(
    server: viser.ViserServer, record_dir_arg: str, record_size: tuple[int, int], mode: str
) -> Path | None:
    """The capture directory, once a browser has connected to render it, or
    ``None`` when not recording. The render comes from the client's three.js
    view, not the server, so recording waits for one."""
    if not record_dir_arg:
        return None
    record_dir = Path(record_dir_arg).expanduser().resolve()
    record_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"  {mode} record: open http://localhost:{server.get_port()} in a browser "
        "to act as the render client; waiting for connection",
        flush=True,
    )
    while not server.get_clients():
        time.sleep(0.5)
    print(
        f"  {mode} record: client connected; capturing PNGs to {record_dir} "
        f"at {record_size[0]}x{record_size[1]}",
        flush=True,
    )
    return record_dir


# ---------------------------------------------------------------------------
# Tour mode (programmatic walk for video capture).
# ---------------------------------------------------------------------------


# Narrative order: easy (EAIK supports) → 6R wedge (EAIK refuses) → 7R
# climax (EAIK refuses entirely). Mesh-rendered arms where available -- the
# primitive skeleton fallback doesn't look cinematic enough for the README
# hero. JACO 2 renders meshes when ``local_urdf_paths`` names a local copy of
# its description.
_TOUR_ORDER: tuple[str, ...] = (
    # Act 1 — easy: EAIK has these.
    "UR5 — three-parallel 6R (Pieper)",
    "Unitree Z1 — three-parallel 6R (UR-class)",
    "Franka Panda — anthropomorphic 7R",
    # Act 2 — wedge: non-Pieper 6R, EAIK refuses.
    "UFactory xArm6 — non-Pieper 6R",
    "Kinova JACO 2 — non-Pieper 6R",
    "AgileX PiPER — non-Pieper 6R",
    # Act 3 — climax: 7R territory, EAIK refuses entirely.
    "KUKA iiwa14 — SRS 7R",
    "Flexiv Rizon 4 — non-SRS 7R",
)


def _lissajous_marker_T(
    T0: np.ndarray,
    t: float,
    duration: float,
    pos_scale: float = 1.0,
) -> np.ndarray:
    """Smooth Lissajous-style oscillation around an anchor pose.

    Returns a 4x4 transform that traces a small ellipsoidal loop in
    position while gently rocking the orientation. Amplitudes are tuned
    so the motion exercises distinct IK branches without ever leaving
    the arm's typical workspace. ``pos_scale`` shrinks the translation
    amplitudes for small arms (e.g. PiPER at ~0.45 m reach) so the
    marker stays in a high-manipulability region rather than skirting
    the workspace boundary.
    """
    phase = 2 * np.pi * t / max(duration, 1e-3)
    # Translation: ellipse in xy + small bob in z. Smaller than before --
    # 10 cm was too much for small arms (PiPER, JACO 2) and visually
    # pushed the marker out of the "obviously reachable" region; 5 cm
    # keeps the active EE comfortably inside the manipulable workspace
    # for every arm in the tour.
    dx = pos_scale * 0.05 * np.sin(phase)
    dy = pos_scale * 0.04 * np.sin(2 * phase + 0.3)
    dz = pos_scale * 0.03 * np.cos(phase) - pos_scale * 0.03
    T = T0.copy()
    T[:3, 3] = T0[:3, 3] + np.array([dx, dy, dz])
    # Rotation: tilt around the marker's local X and Y by small angles
    # over the loop. Keeps wrist branches visibly different per pose.
    ax = 0.35 * np.sin(phase + 0.5)
    ay = 0.25 * np.cos(phase * 0.8)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(ax), -np.sin(ax)],
            [0, np.sin(ax), np.cos(ax)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(ay), 0, np.sin(ay)],
            [0, 1, 0],
            [-np.sin(ay), 0, np.cos(ay)],
        ]
    )
    T[:3, :3] = T0[:3, :3] @ Rx @ Ry
    return T


def _make_capture(server, record_dir: Path | None, record_size: tuple[int, int]):
    """Build the per-frame PNG capture used by the recorded modes.

    Pulls a server-rendered frame from the first connected client and writes
    it zero-padded, returning the next frame index. A no-op (returns the index
    unchanged) when not recording or when no client has connected, so the same
    call site works whether or not a browser is attached.
    """

    def _capture(frame_idx: int) -> int:
        if record_dir is None or server is None:
            return frame_idx
        clients = server.get_clients()
        if not clients:
            return frame_idx
        client = next(iter(clients.values()))
        try:
            import imageio.v3 as iio  # type: ignore[import-not-found]

            img = client.get_render(
                height=record_size[1], width=record_size[0], transport_format="jpeg"
            )
            iio.imwrite(record_dir / f"frame_{frame_idx:05d}.png", img)
        except Exception as e:
            print(f"  record: capture failed at frame {frame_idx}: {e}", flush=True)
        return frame_idx + 1

    return _capture


def _run_tour(
    *,
    select_arm,
    arm_dropdown,
    max_ghosts_slider,
    move_marker,
    solve_and_render,
    state,
    lock,
    per_arm_s: float,
    max_ghosts: int,
    server=None,
    record_dir: Path | None = None,
    record_size: tuple[int, int] = (1280, 720),
    settle_after_load_s: float = 2.5,
) -> None:
    """Drive the GUI through ``_TOUR_ORDER``: switch arm, hold the home
    pose briefly for the mesh upload + viewer settle, then oscillate
    the marker for ``per_arm_s`` seconds at ~30 fps.

    ``marker.on_update`` only fires on client→server traffic, so the
    tour explicitly calls ``solve_and_render`` after every marker pose
    update -- otherwise the arm would freeze at its home pose while the
    marker animates silently.
    """
    fps = 30
    arm_labels = {spec.label for spec in ARMS}

    _capture = _make_capture(server, record_dir, record_size)

    # Per-arm frame ranges, written to ``_manifest.json`` so a downstream
    # ffmpeg pass can carve out each arm's GIF without re-running the tour.
    frame_ranges: dict[str, dict[str, int | str]] = {}
    frame_idx = 0
    for label in _TOUR_ORDER:
        if label not in arm_labels:
            print(f"  tour: skipping unknown arm {label!r}", flush=True)
            continue
        t_select_0 = time.perf_counter()
        # Setting the dropdown is cosmetic and also fires its on_update, which
        # loads the arm; select_arm is idempotent, so calling it here as well
        # only guarantees the arm is loaded before the motion starts.
        arm_dropdown.value = label
        select_arm(label)
        arm_start_idx = frame_idx
        # Cap the visible-ghost count: 32-branch arms otherwise spike to
        # ~1s render frames on the heavy 7Rs (Rizon 4 / Kassow). The full
        # branch count still appears in the stats badge; we just don't
        # paint every ghost. The active arm is unaffected.
        max_ghosts_slider.value = min(max_ghosts, int(max_ghosts_slider.max))
        t_select = time.perf_counter() - t_select_0
        # Anchor pose: the rendered EE of the freshly-loaded arm.
        with lock:
            rt: ArmRuntime = state["arm"]
            T_anchor = rt.active.base_offset @ rt.arm.fk(rt.tracker.q)
        # Per-arm position-amplitude scale: bigger reach → bigger Lissajous.
        # ``reach`` here is the home-pose EE distance from the base, a
        # rough proxy. Cap at 1.4 so PiPER (~0.45 m) gets a smaller loop
        # than Rizon 4 (~1.0 m) without making the motion invisible.
        reach = float(np.linalg.norm(T_anchor[:3, 3]))
        pos_scale = max(0.6, min(1.4, reach / 0.6))
        # Per-arm cinematic camera: orbit ~60° off-axis at ~25° elevation
        # so we see the wrist branches splay out clearly. Distance scales
        # with reach so the arm fills the frame. Look-at is the home EE
        # so the marker stays roughly centered through the motion.
        if server is not None and record_dir is not None:
            clients = server.get_clients()
            if clients:
                cam = next(iter(clients.values())).camera
                look_at = T_anchor[:3, 3].copy()
                # Aim down toward the arm rather than at the EE -- frames
                # the whole kinematic chain instead of only the wrist.
                look_at[2] = max(0.0, look_at[2] - 0.15 * reach)
                cam_distance = max(1.2, reach * 2.3)
                az, el = np.radians(60), np.radians(20)
                cam.position = look_at + cam_distance * np.array(
                    [
                        np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az),
                        np.sin(el),
                    ]
                )
                cam.look_at = look_at
                cam.fov = np.radians(38)  # mild telephoto for "cinematic"
        # Force the client to fully process the arm-switch queue before we
        # start the motion loop and capturing. ``flush`` pushes buffered
        # messages; the sacrificial ``get_render`` is a real sync barrier
        # -- it blocks until the client has rendered the new scene, which
        # only happens after every queued mesh remove + new mesh add has
        # been applied. Without this barrier the first ~N captured frames
        # of the new arm can still show residual meshes from the previous
        # arm (the "two arms in one GIF" symptom).
        if server is not None and record_dir is not None:
            clients = server.get_clients()
            if clients:
                client = next(iter(clients.values()))
                client.flush()
                with contextlib.suppress(Exception):
                    _ = client.get_render(height=128, width=128, transport_format="jpeg")
        # Mesh upload over WebSocket can stretch to ~1s on big-mesh arms
        # (Panda, iiwa, FANUC CRX). Hold the home pose for a beat so the
        # viewer sees the arm settled before the marker starts moving --
        # AND so the previous arm's ``remove()`` messages have time to
        # drain on the client before the new mesh tree arrives. Without
        # this gap the residual nodes-pending-deletion compound across
        # arms and the browser-side scene grows unboundedly.
        time.sleep(settle_after_load_s)
        n_frames = max(int(per_arm_s * fps), 1)
        t_motion_0 = time.perf_counter()
        frame_times: list[float] = []
        for k in range(n_frames):
            t_frame_0 = time.perf_counter()
            t = k / fps
            T = _lissajous_marker_T(T_anchor, t, per_arm_s, pos_scale=pos_scale)
            with lock:
                move_marker(T)
                had_sols = solve_and_render()
            # Only capture frames where the tracker followed the marker.
            # Otherwise the recording would freeze the arm at its last
            # reachable pose while the marker keeps moving -- visually
            # identical to a stutter / dead frame.
            if had_sols:
                frame_idx = _capture(frame_idx)
            frame_times.append(time.perf_counter() - t_frame_0)
            # When NOT recording: pace to wall-clock so live playback
            # honors the chosen per_arm_s budget. When recording: skip
            # the sleep -- each captured PNG IS one output frame, the
            # final 30fps video is built by ffmpeg from the PNG sequence
            # regardless of how long each capture took.
            if record_dir is None:
                target = t_motion_0 + t
                now = time.perf_counter()
                if target > now:
                    time.sleep(target - now)
        median_ms = sorted(frame_times)[len(frame_times) // 2] * 1000
        max_ms = max(frame_times) * 1000
        print(
            f"  tour: {label}  "
            f"(select={t_select * 1000:.0f}ms  "
            f"frame median={median_ms:.0f}ms  max={max_ms:.0f}ms)",
            flush=True,
        )
        # Match the label back to its module_name (artifact module) so
        # downstream encoders produce GIFs named after the ssik artifact.
        spec = next((a for a in ARMS if a.label == label), None)
        if spec is not None and record_dir is not None and frame_idx > arm_start_idx:
            frame_ranges[spec.module_name] = {
                "label": label,
                "start": arm_start_idx,
                "end_exclusive": frame_idx,
            }
    if record_dir is not None and frame_ranges:
        import json

        (record_dir / "_manifest.json").write_text(
            json.dumps({"frame_ranges": frame_ranges, "fps": fps}, indent=2)
        )


def _uniform_q(kb, rng) -> np.ndarray:
    """A random configuration, using +/-pi for joints with no limits."""
    out = []
    for j in kb.joints:
        lo, hi = j.limits if j.limits is not None else (-np.pi, np.pi)
        out.append(rng.uniform(lo, hi))
    return np.array(out, dtype=float)


def _pick_self_motion_branch(arm, kb, *, rng, tries: int = 12, min_points: int = 40):
    """A pose and branch worth animating, from ``tries`` random draws.

    Scored by how far the *proximal* joints travel along the branch, not by
    arc length. Arc length counts wrist rotation equally with shoulder
    rotation, and a branch that spends its length spinning joint 7 barely
    moves on screen: an early draft picked one of those and changed 2.5% of
    the pixels across the whole sweep. Shoulder and elbow travel is what a
    viewer actually sees, and it is also the honest illustration, since the
    point being made is that the arm reconfigures substantially while the
    hand holds still.
    """
    best = None
    for _ in range(tries):
        t_target = arm.fk(_uniform_q(kb, rng))
        try:
            manifold = arm.self_motion(t_target)
        except Exception as e:  # a draw near a singularity is not worth animating
            print(f"  self-motion: skipping a pose ({type(e).__name__})", flush=True)
            continue
        for chart in manifold.charts:
            segments = chart.sample(400, limits=True)
            if not segments:
                continue
            _, qs = max(segments, key=lambda s: len(s[0]))
            if len(qs) < min_points:
                continue
            travel = float(np.sum(qs[:, :4].max(axis=0) - qs[:, :4].min(axis=0)))
            if best is None or travel > best[0]:
                best = (travel, t_target, len(manifold.charts), qs)
    return None if best is None else best[1:]


def _run_self_motion(
    *,
    select_arm,
    move_marker,
    state,
    lock,
    arm_label: str,
    seconds: float,
    n_ghosts: int = 0,
    server=None,
    record_dir: Path | None = None,
    record_size: tuple[int, int] = (1280, 720),
    settle_s: float = 3.0,
    seed: int = 23,
) -> None:
    """Sweep one branch of a redundant arm's self-motion manifold with the
    end-effector pinned in place.

    The animation makes the feature's claim visually: the marker never moves,
    the ghosts are frozen postures along the same continuous branch, and the
    solid arm slides through them. Every frame is an exact IK solution for one
    unchanging target, not an interpolation between two of them, and the
    printed EE drift is the evidence.
    """
    fps = 30
    _capture = _make_capture(server, record_dir, record_size)

    select_arm(arm_label)
    with lock:
        rt: ArmRuntime | None = state["arm"]
    if rt is None:
        raise SystemExit(f"self-motion: could not load {arm_label!r}")

    arm = rt.arm
    kb = arm.kinbody
    if arm.dof < 7:
        raise SystemExit(
            f"self-motion needs a redundant arm; {arm_label!r} has {arm.dof} DOF. "
            f"Pick a 7R, e.g. --self-motion-arm 'Franka Panda — anthropomorphic 7R'"
        )

    print(f"  self-motion: searching for a long admissible arc on {arm_label}", flush=True)
    picked = _pick_self_motion_branch(arm, kb, rng=np.random.default_rng(seed))
    if picked is None:
        raise SystemExit("self-motion: no branch with a usable in-limits arc; try another --seed")
    t_target, n_charts, qs = picked

    # The claim, measured before a single frame is drawn. If this is not at
    # FK tolerance the animation would be showing something untrue.
    drift = max(float(np.linalg.norm(arm.fk(q) - t_target)) for q in qs)
    print(
        f"  self-motion: {n_charts} branches at this pose; animating {len(qs)} postures "
        f"along one of them, EE drift <= {drift:.2e}",
        flush=True,
    )
    if drift > 1e-9:
        raise SystemExit(f"self-motion: postures do not hold the pose (drift {drift:.2e})")

    # Out along the arc and back, so the GIF loops without a jump cut.
    sweep = np.vstack([qs, qs[::-1][1:-1]]) if len(qs) > 2 else qs

    # Ghosts are frozen waypoints on the same arc: the continuum the solid arm
    # is sliding along. They are set once and never touched again, which is
    # why this drives the renderers directly instead of going through
    # ``_solve_and_render`` (that re-solves and rebinds every ghost per frame).
    # The lock is held for the whole sweep so a stray GUI event cannot redraw
    # the arms in between.
    with lock:
        slots = min(n_ghosts, len(rt.ghosts))
        picks = np.linspace(0, len(qs) - 1, slots).astype(int) if slots else np.array([], dtype=int)
        move_marker(rt.active.base_offset @ t_target)
        with server.atomic():
            for slot in range(slots):
                rt.ghosts[slot].set_visible(True)
                rt.ghosts[slot].set_q(qs[picks[slot]])
            for slot in range(slots, len(rt.ghosts)):
                rt.ghosts[slot].set_visible(False)
            rt.active.set_q(sweep[0])

        # Frame the shot. The interactive default camera is wherever the user
        # last orbited to; a captured asset needs a deliberate viewpoint, or
        # the arm renders as a speck in the middle of an empty frame.
        # Three-quarter view from slightly above, framed on the target the arm
        # is holding.
        target = (rt.active.base_offset @ t_target)[:3, 3]
        reach = float(np.linalg.norm(target)) or 0.8
        # Frame on the arm's mid-height rather than the target itself: the
        # sweep moves the elbow far more than the hand, and the hand is by
        # construction the one thing that does not move.
        base = rt.active.base_offset[:3, 3]
        focus = 0.5 * (base + target) + np.array([0.0, 0.0, 0.08])
        for client in (server.get_clients() or {}).values():
            with contextlib.suppress(Exception):
                client.camera.position = focus + reach * np.array([0.90, -0.75, 0.30])
                client.camera.look_at = focus
                client.camera.up_direction = np.array([0.0, 0.0, 1.0])

        # Let the meshes upload and the viewer settle before the first
        # captured frame, then force one round-trip so the settle actually
        # completed.
        time.sleep(settle_s)
        if server is not None and record_dir is not None:
            clients = server.get_clients()
            if clients:
                with contextlib.suppress(Exception):
                    _ = next(iter(clients.values())).get_render(
                        height=128, width=128, transport_format="jpeg"
                    )

        # Map the requested duration onto the whole out-and-back loop.
        # Indexing ``sweep`` by frame number instead would make the fraction
        # of the arc covered depend on the frame count: at 30fps a 3s capture
        # would walk 90 of this branch's 800 postures and the arm would look
        # almost still.
        n_frames = max(int(seconds * fps), 2)
        walk = np.linspace(0.0, len(sweep) - 1.0, n_frames).round().astype(int)

        frame_idx = 0
        for i in range(n_frames):
            with server.atomic():
                rt.active.set_q(sweep[walk[i]])
            frame_idx = _capture(frame_idx)
            if record_dir is None:
                time.sleep(1.0 / fps)

    if record_dir is not None:
        import json

        (record_dir / "_manifest.json").write_text(
            json.dumps(
                {
                    "frame_ranges": {
                        "self_motion": {
                            "label": arm_label,
                            "start": 0,
                            "end_exclusive": frame_idx,
                        }
                    },
                    "fps": fps,
                },
                indent=2,
            )
        )
        print(f"  self-motion: captured {frame_idx} frames to {record_dir}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to serve on (default 127.0.0.1, this machine only; "
        "0.0.0.0 serves the whole network)",
    )
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument(
        "--no-meshes",
        action="store_true",
        help="render every arm as kinematic primitives: no description downloads, "
        "so it runs offline and headless",
    )
    parser.add_argument(
        "--tour",
        action="store_true",
        help="run a scripted tour through the arm roster (for video capture)",
    )
    parser.add_argument(
        "--tour-delay",
        type=float,
        default=4.0,
        help="seconds to wait after server start before the tour begins "
        "(gives a screen recorder time to attach)",
    )
    parser.add_argument(
        "--tour-per-arm",
        type=float,
        default=3.0,
        help="seconds spent on each arm during the tour",
    )
    parser.add_argument(
        "--tour-settle",
        type=float,
        default=2.5,
        help="seconds to hold each arm still after it loads, so meshes finish "
        "uploading to the viewer before the motion (and capture) starts",
    )
    parser.add_argument(
        "--tour-exit",
        action="store_true",
        help="exit cleanly after the tour completes (default: keep server running)",
    )
    parser.add_argument(
        "--tour-max-ghosts",
        type=int,
        default=7,
        help="cap how many ghost branches render per arm during the tour "
        "(default 7 -- visually balanced + keeps heavy 7Rs within frame budget); "
        "the stats badge still shows the full branch count",
    )
    parser.add_argument(
        "--tour-record-dir",
        default="",
        help="if set, capture every tour frame as a PNG into this directory. "
        "Requires a connected browser client (open the demo URL in any tab) "
        "and ``imageio`` on the runtime path. Tour skips wall-clock pacing "
        "while recording; ffmpeg later composes the PNGs into a 30fps MP4.",
    )
    parser.add_argument(
        "--tour-record-size",
        default="1280x720",
        help="WxH of captured PNGs (default 1280x720). 1920x1080 also works.",
    )
    parser.add_argument(
        "--self-motion",
        action="store_true",
        help="animate one branch of a redundant arm's self-motion manifold with "
        "the end-effector pinned (for the README asset). Reuses --tour-record-dir, "
        "--tour-record-size, --tour-delay and --tour-settle.",
    )
    parser.add_argument(
        "--self-motion-arm",
        default="Franka Panda — anthropomorphic 7R",
        help="which arm to sweep; must be 7-DOF (default: the Panda)",
    )
    parser.add_argument(
        "--self-motion-seconds",
        type=float,
        default=8.0,
        help="length of the sweep in seconds at 30fps (default 8)",
    )
    parser.add_argument(
        "--self-motion-seed",
        type=int,
        default=23,
        help="seed for the pose search; change it to get a different branch",
    )
    parser.add_argument(
        "--self-motion-ghosts",
        type=int,
        default=0,
        help="frozen waypoint arms along the same branch (default 0). Ghosts "
        "render in the same color as the active arm, which reads as a fan of "
        "peers rather than a trail, so the sweep is clearer without them.",
    )
    args = parser.parse_args()
    w, h = (int(s) for s in args.tour_record_size.lower().split("x"))
    main(
        host=args.host,
        port=args.port,
        meshes=not args.no_meshes,
        tour=args.tour,
        tour_delay_s=args.tour_delay,
        tour_per_arm_s=args.tour_per_arm,
        tour_settle_s=args.tour_settle,
        tour_exit=args.tour_exit,
        tour_max_ghosts=args.tour_max_ghosts,
        tour_record_dir=args.tour_record_dir,
        tour_record_size=(w, h),
        self_motion=args.self_motion,
        self_motion_arm=args.self_motion_arm,
        self_motion_seconds=args.self_motion_seconds,
        self_motion_seed=args.self_motion_seed,
        self_motion_ghosts=args.self_motion_ghosts,
    )
