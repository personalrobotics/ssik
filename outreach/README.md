# Outreach drafts

Draft copy + recipes for v1.1 adoption push. Nothing here is part of the published package; everything is for posting / submitting under Siddhartha's accounts.

## Contents

- `zenodo-setup.md` — one-time GitHub-Zenodo integration. ~10 minutes; requires Siddhartha's Zenodo + GitHub login.
- `reddit-robotics.md` — draft post for `r/robotics`.
- `ros-discourse.md` — draft post for `https://discourse.ros.org` (Category: General).
- `robotics-worldwide.md` — draft post for the `robotics-worldwide@usc.edu` mailing list.
- `conda-forge/` — staged-recipes draft + submission instructions.

## Suggested order

1. **Zenodo first** — flip the toggle now, then on the next tagged release the DOI gets minted automatically. Costs 10 minutes; unblocks "ssik has a DOI" in any later post.
2. **Conda-forge second** — PR into `conda-forge/staged-recipes`. Review takes ~1 week of back-and-forth with conda-forge maintainers; getting it submitted early means it's live by the time the Reddit / ROS Discourse posts go up.
3. **Announcements third** — once the DOI exists and conda install works, the posts cite both, which is a better story than "PyPI only, citation pending."
