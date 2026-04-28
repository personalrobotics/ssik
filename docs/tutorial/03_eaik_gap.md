# 3. The EAIK gap

!!! warning "Scaffolding"
    Outline below; prose to be filled in.

## What this chapter covers

The arms whose IK is **not** a one-line call against EAIK or IK-Geo, and why the obvious workarounds don't work.

### Arms outside the Pieper class

- **Kinova JACO 2 (j2n6s200):** 60° non-orthogonal twists at joints 4–5. The quat=(0, 0, 0.5, 0.866) rotation between consecutive wrist links means no two wrist axes intersect, no axis triple is parallel, and the standard subproblem compositions all fail their preconditions. Real fixture in [`tests/fixtures/jaco2.py`](https://github.com/siddhss5/ikfastpy/blob/main/tests/fixtures/jaco2.py), transcribed from `robot-code/ada_assets/.../jaco2.xml` MJCF (#80).
- **Agilex Piper:** similar non-Pieper geometry.
- **Flexiv Rizon 4:** 7-DOF non-SRS arm. Joint-locking turns it into a 6R per lock value, but the resulting sub-chain is non-Pieper at every lock value.
- **Custom geometries** from machined-from-scratch arms or kinematic prototyping.

### Why the obvious approaches don't work

- **Subproblem composition (EAIK / IK-Geo):** the geometric specialisations (three intersecting wrist axes, three parallel shoulder axes) never apply. SP5's polynomial setup has structural degeneracy on these chains.
- **IKFast (Diankov 2010):** its symbolic codegen pipeline depends on a 2010-vintage sympy that doesn't exist anymore. Modern sympy + mpmath stack causes `polyroots NoConvergence` at substitution / `solveLiWoernleHiller` time. Verified on real arms — UR5 (797s before failure on cambel URDF), Puma-with-d5=0.01 (617s, NoConvergence), JACO 2. Hours-to-days of derivation on cases where it works at all. The vendored `_legacy/` tree is a museum piece.
- **Numeric IK (mink, KDL, TRAC-IK):** works but pays the iterative cost, returns one solution rather than the full branch set, and gives no up-front "unreachable" signal. Mink at ~20 ms per call versus ssik's ~2.25 ms median on JACO 2 (post-Tier 2.3 of #86).

### What this means for ssik's design

- **ssik bundles tier-0/1 closed-form solvers** for the Pieper-class arms (covered by EAIK / IK-Geo), so you don't pay the tier-2 cost when you don't need to.
- **ssik's tier-2 numeric Raghavan–Roth solver** ([Chapter 4](04_raghavan_roth.md)) handles the gap arms.
- **The dispatcher** picks the right tier per kb at registration time; users call `solve(kb, T)` and don't think about it.

### The strategic frame

EAIK / IK-Geo cover the easy 80% of commercial arms. ssik covers the harder 20% — the EAIK gap. That's where a millisecond-level analytical IK solution didn't exist before. Memory entry [`project_eaik_gap_strategy`](https://github.com/siddhss5/ikfastpy/issues/78) covers the strategic positioning.

## References

- Raghavan, M. & Roth, B. (1990). "Inverse kinematics of the general 6R manipulator and related linkages." *J. Mech. Design*.
- Manocha, D. & Canny, J. F. (1994). "Efficient inverse kinematics for general 6R manipulators." *IEEE T-RA* 10(5):648–657.
- Tsai, L.-W. (1999). *Robot Analysis: The Mechanics of Serial and Parallel Manipulators.* Wiley. Appendix C.
