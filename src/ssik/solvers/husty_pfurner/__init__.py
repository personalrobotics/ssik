"""Husty-Pfurner universal 6R / 6R-P analytical IK (Phase 5 of GitHub #158).

Implementation in progress, tracked in #162.

The runtime entry point is :func:`ssik.solvers.husty_pfurner.general_6r.solve`.
Until the algorithm lands it raises :class:`NotImplementedError`. The validation
harness in ``tests/test_husty_pfurner_oracles.py`` is wired up and runs against
this skeleton (all gates ``xfail(strict=True)``); the gates flip to plain
passing tests when the solver is implemented.

Algorithmic reference: Capco, Loquias, Manongsong, Nemenzo (2019),
'Inverse Kinematics of Some General 6R/P Manipulators', arXiv 1906.07813.
Builds on Husty, Pfurner, Schröcker (2007) MMT 42(1):66-81.
"""
