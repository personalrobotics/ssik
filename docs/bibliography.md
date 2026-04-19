# Bibliography

Primary sources and recommended further reading.

## Primary source

**Diankov, R.** (2010). *Automated Construction of Robotic Manipulation Programs.* PhD thesis, CMU-RI-TR-10-29, Carnegie Mellon University. [PDF](http://www.programmingvision.com/rosen_diankov_thesis.pdf).

> Primary source for the IKFast algorithm. Section 4.1 (pp. 78–97) covers the symbolic IK derivation that this package re-implements, with subsections dedicated to 3D translation IK (§4.1.3), 3D rotation IK (§4.1.4), 6D transformation IK (§4.1.5), 4D ray IK (§4.1.6), and the handling of redundant joints (§4.1.7). Section 3.5 (pp. 57–60) covers free-joint planning, which is the foundation of the locked-joint approach to redundant manipulators (e.g. Franka, KUKA iiwa).

## Foundational IK literature

To be filled in by chapter authors as they cite. Expected entries:

- **Pieper, D. L.** (1968). *The Kinematics of Manipulators Under Computer Control.* PhD thesis, Stanford University. — The classical solvability condition for 6-DOF arms with three intersecting axes (spherical wrist).
- **Paden, B.** (1986). *Kinematics and Control of Robot Manipulators.* PhD thesis, UC Berkeley. — Subproblem decomposition for closed-form IK.
- **Manocha, D. & Canny, J.** (1994). Efficient inverse kinematics for general 6R manipulators. *IEEE Transactions on Robotics and Automation*, 10(5). — The algebraic elimination strategy underlying IKFast's resultant approach.
- **Murray, R., Li, Z., Sastry, S.** (1994). *A Mathematical Introduction to Robotic Manipulation.* CRC Press. — Standard reference for Product-of-Exponentials and the geometric foundations.

## Comparable software

- **EAIK** (Ostermeier et al.). [GitHub](https://github.com/OstermD/EAIK). Family-detection-based analytic IK; complementary to IKFast's per-robot symbolic derivation. Discussed in Tutorial Chapter 6.
- **TRAC-IK** (Beeson & Ames, 2015). Numerical IK with augmented strategies for singularities and joint-limit corner cases.
- **KDL** (Orocos). The numerical-IK reference in the ROS ecosystem.
