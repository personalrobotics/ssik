# IK modes

IKFast supports 16 distinct IK modes, each constraining a different combination of end-effector position and orientation. Most users have only ever encountered `Transform6D` (full 6-DOF pose), but the other modes are well-suited to specific task families: welding seams (`TranslationDirection5D`), drilling holes (`TranslationZAxisAngle4D`), camera pointing (`Lookat3D`), planar pick-and-place (`TranslationXYOrientation3D`), and so on.

!!! note "Under construction"
    Per-mode pages are tracked in issue [#18](https://github.com/siddhss5/ikfastpy/issues/18). Each will include the constraint surface, geometric figure, mathematical parametrization, real-world task example, code snippet, and a citation to the corresponding section of Diankov's thesis where applicable.

## The 16 modes

| Mode | Target DOF | Typical task |
|---|---|---|
| `Transform6D` | 6 | General grasping (full pose) |
| `Translation3D` | 3 | Painting, cleaning |
| `Rotation3D` | 3 | Camera aiming (orientation only) |
| `Direction3D` | 2 | Sweeping a pointer |
| `Ray4D` | 5 | Drilling, screwing |
| `Lookat3D` | 3 | Sensor / camera pointing |
| `TranslationDirection5D` | 5 | Welding seams |
| `TranslationXY2D` | 2 | SCARA-style tasks |
| `TranslationXYOrientation3D` | 3 | Planar pick-and-place |
| `TranslationLocalGlobal6D` | 6 | Local point on EE → global point |
| `TranslationXAxisAngle4D` | 4 | Translation + roll about X |
| `TranslationYAxisAngle4D` | 4 | Translation + roll about Y |
| `TranslationZAxisAngle4D` | 4 | Translation + roll about Z |
| `TranslationXAxisAngleZNorm4D` | 4 | Translation + EE-X angle from Z normal |
| `TranslationYAxisAngleXNorm4D` | 4 | Translation + EE-Y angle from X normal |
| `TranslationZAxisAngleYNorm4D` | 4 | Translation + EE-Z angle from Y normal |
