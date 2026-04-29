"""Symbolic versions of the SP1-SP6 closed-form subproblem solvers.

Each module here mirrors :mod:`ssik.subproblems.spN` but takes sympy
:class:`Matrix` / :class:`Symbol` / :class:`Float` inputs and returns
sympy expressions for the solution(s). Substituting concrete arm
constants and running :func:`sympy.cse` produces the inlined trig
output the codegen emits.

Implemented subset (this slice covers Puma's spherical_two_parallel
end-to-end):

- :mod:`.sp1` -- SP1 (rotate p to match q): single ``atan2`` expression.
- :mod:`.sp4` -- SP4 (project rotated p onto h): ``atan2`` + ``acos``,
  two-branch.
- :mod:`.sp3` -- SP3 (rotate p so distance from q equals d): reduces
  to SP4 with a target shift.

Tests assert the symbolic outputs evaluate to the same numeric values
as the runtime SP solvers on a few sample inputs.
"""
