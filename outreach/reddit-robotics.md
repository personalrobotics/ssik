# Draft: r/robotics post

**Subreddit:** r/robotics
**Suggested flair:** Software / Open Source
**Suggested title:** `ssik: analytical IK for 6R/7R arms — every branch, machine-precision FK, pip install`

---

## Title

`ssik: analytical IK for 6R/7R arms — every branch, machine-precision FK, pip install`

## Body

I've open-sourced **ssik**, a Python library that returns every analytical inverse-kinematics branch for 6R and 7R robot arms.

```bash
pip install ssik
```

```python
from ssik.prebuilt import franka_panda_ik
import numpy as np

T = np.eye(4); T[:3, 3] = [0.5, 0.1, 0.3]
sols = franka_panda_ik.solve(T)    # every IK branch
```

Each `Solution` carries the joint vector, the FK residual against the target, and which polish path fired. Empty list = pose is unreachable.

**Arms shipped:** UR5, Puma 560, Kinova JACO 2, KUKA iiwa14, Kinova Gen3, Franka Panda, Flexiv Rizon 4, Kassow KR810, UFactory xArm7, UFactory xArm6, Unitree Z1, AgileX PiPER. For anything else: `ssik build my_arm.urdf` emits a single-file Python artifact for your URDF.

**What's different from existing tools:**
- vs **numerical IK** (TracIK, MINK, KDL): those run damped least-squares to *one* converged config and stop. ssik enumerates every analytical branch — important for motion planning (try every branch, pick best clearance), dexterity analysis (per-branch manipulability), trajectory continuation across singularities.
- vs **IKFast**: same per-arm specialised codegen idea, but ssik solves arms IKFast can't (non-Pieper 6R like JACO 2's 55° twists). Pure Python at runtime, no C++ codegen.
- vs **IK-Geo / EAIK**: those refuse non-Pieper 6R and non-SRS 7R. ssik dispatches to those libraries' subproblem decomposition where it applies, falls back to a Husty-Pfurner Study-quaternion solver where it doesn't.

**Honest perf numbers:** ~0.2-5 ms/IK for closed-form classes (UR5, iiwa14), ~25-200 ms/IK for the non-SRS 7R fallback path (Rizon 4, Kassow). FK closure ~1e-8 default, tightenable to ~1e-12 via the `TolerancePolicy` knobs.

**Repo:** https://github.com/personalrobotics/ssik
**Docs:** https://personalrobotics.github.io/ssik/
**DOI:** https://doi.org/10.5281/zenodo.20278005
**License:** BSD-3-Clause. Clean-room reimplementations from the academic literature; lineage documented per-module.

Happy to answer questions about the algorithmic side or how it compares to whatever you're using.

---

## Notes for posting

- Subreddit rules ban self-promotion that isn't clearly contributory. The angle "I open-sourced this, technical Q&A welcome" reads fine; "go upvote my PyPI" does not.
- Best post times: weekday mornings US Pacific (broad audience awake worldwide).
- Reply to early comments quickly — engagement in the first hour heavily weights ranking.
- If asked "why not just use EAIK / IK-Geo": answer is the non-Pieper 6R + non-SRS 7R + approximate-SRS Gen3 coverage gap. Do not bash competitors.
