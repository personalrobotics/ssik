"""Codegen: per-arm IK artifact emission.

Two emission modes:

* **Thin wrapper** (#110 Phase 1): the artifact's ``solve()`` calls into
  the generic ssik solver at runtime. KinBody constants are baked, but
  the math lives in :mod:`ssik.solvers`. Used for tier-2 today.

* **Specialised** (#112): the artifact's ``solve()`` body inlines the
  algebraic IK as straight-line trig + arithmetic with the arm's
  constants substituted. Tier-0 only in this slice. Used for any plan
  whose solver has a registered symbolic composer (see
  :mod:`ssik.codegen._compose`).

The public surface :func:`ssik.core.codegen.emit_artifact` picks between
them based on ``DispatchPlan.tier`` and the registered composers.
"""
