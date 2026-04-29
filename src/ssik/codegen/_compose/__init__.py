"""Per-solver composers: build the full symbolic algebraic chain.

Each composer takes:

* a POE-normalised :class:`~ssik._kinbody.KinBody` whose constants get
  substituted into the symbolic SP solvers,
* sympy symbols for the target-pose entries (``r_00 .. r_22``,
  ``p_x``, ``p_y``, ``p_z``).

It returns a :class:`ComposedSolver`: the symbolic q-vector candidate
expressions, plus the runtime "guards" (LS feasibility checks,
degeneracy conditions, branch enumerations) that the codegen needs to
emit alongside the inlined trig.

The codegen then runs :func:`sympy.cse` over the candidate expressions,
emits a Python source file with the explicit math, and pairs it with the
runtime control-flow (which `if`-blocks, which `for`-loops over branches).
"""
