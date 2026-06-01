# Draft: r/robotics post

**Subreddit:** r/robotics
**Suggested flair:** Software / Open Source
**Suggested title:** `ssik: analytical IK for 6R/7R arms — every branch, machine-precision FK, pip install`

---

## Title

`ssik: analytical IK for 6R/7R arms — every branch, machine-precision FK, pip install`

## Body

Hi r/robotics — I've open-sourced **ssik**, a Python library that returns every analytical inverse-kinematics branch for 6R and 7R robot arms. Posting in case it's useful to anyone here; happy to help anyone get it working on their setup.

```bash
pip install ssik
```

```python
from ssik.prebuilt import franka_panda_ik
import numpy as np

T = np.eye(4); T[:3, 3] = [0.5, 0.1, 0.3]
sols = franka_panda_ik.solve(T)    # every IK branch
```

Each `Solution` carries the joint vector, the FK residual against the target, and which polish path fired. Empty list = pose is unreachable. If you want to know *why* a pose came back empty, `solve(T, explain=True)` gives you a structured trace of which dispatch tier ran, which subproblem failed, and the residuals — really helpful for debugging.

### A bit of history

ssik comes out of the Personal Robotics Lab, and it's the successor to Rosen Diankov's original IKFast. IKFast has been the gold standard for analytical IK for over a decade. ssik is meant to extend that lineage to the kinematic classes IKFast's pipeline never quite handled well, in a "just pip install it" form factor.

### Arms shipped

UR5, Puma 560, Kinova JACO 2, KUKA iiwa14, Kinova Gen3, Franka Panda, Flexiv Rizon 4, Kassow KR810, UFactory xArm7, UFactory xArm6, Unitree Z1, AgileX PiPER, Flexiv Rizon 10. Every one of them gets a 500-pose Hypothesis fuzz sweep on every PR — the goal is "correct, on every pose, every time", and if you find one it gets wrong I'd genuinely like to hear about it.

For anything else: `ssik build my_arm.urdf` emits a single-file Python artifact for your URDF. If your favourite arm isn't on the list, open an issue with the URDF and I'll happily help add it.

### What's different from existing tools

ssik is meant to complement, not replace, what's already out there.

- vs **numerical IK** (TracIK, MINK, KDL): those run damped least-squares to *one* converged config and stop. Excellent for what they do. ssik enumerates every analytical branch — which is what you actually want for motion planning (try every branch, pick best clearance), dexterity analysis (per-branch manipulability), trajectory continuation across singularities, and — the use case I'm most excited about — **teleop pipelines for demonstration collection**, where `max_solutions=1, q_seed=q_current` gives jump-free joint trajectories at controller rate for imitation-learning / VLA data rigs.
- vs **IKFast**: same per-arm specialised codegen idea — and the direct ancestor in spirit. ssik adds non-Pieper 6R (e.g. JACO 2's 60° twists) and non-SRS 7R coverage, and runs pure Python at runtime with no C++ codegen step. For arms IKFast handles well, IKFast is still a great choice.
- vs **IK-Geo / EAIK**: those refuse non-Pieper 6R and non-SRS 7R by design. ssik dispatches to subproblem decomposition where it applies and falls back to a Husty-Pfurner Study-quaternion solver where it doesn't — so you get analytical coverage across the whole tier list.

### Perf numbers

~0.2-5 ms/IK for closed-form classes (UR5, iiwa14), ~25-200 ms/IK for the non-SRS 7R fallback path (Rizon 4, Kassow). FK closure ~1e-8 default, tightenable to ~1e-12 via the `TolerancePolicy` knobs.

### Links

- **Repo:** https://github.com/personalrobotics/ssik
- **Docs:** https://personalrobotics.github.io/ssik/
- **DOI:** https://doi.org/10.5281/zenodo.20278005
- **License:** BSD-3-Clause. Clean-room reimplementations from the academic literature; lineage documented per-module.

Happy to answer questions about the algorithmic side, comparisons to whatever you're using, or how to get it working on a specific arm. Thanks to everyone here who's contributed to the open robotics-software ecosystem over the years — this project stands on a lot of shoulders.

— Siddhartha Srinivasa
Professor, Paul G. Allen School of Computer Science and Engineering
Personal Robotics Laboratory, University of Washington
https://goodrobot.ai

---

## Notes for posting

- Subreddit rules ban self-promotion that isn't clearly contributory. The angle "I open-sourced this, technical Q&A welcome" reads fine; "go upvote my PyPI" does not.
- Best post times: weekday mornings US Pacific (broad audience awake worldwide).
- Reply to early comments quickly — engagement in the first hour heavily weights ranking.
- If asked "why not just use EAIK / IK-Geo": answer is the non-Pieper 6R + non-SRS 7R + approximate-SRS Gen3 coverage gap. Do not bash competitors.
