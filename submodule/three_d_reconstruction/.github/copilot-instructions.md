## Purpose
This repository contains small tooling for reconstructing 3D points from omnidirectional camera observations. Keep changes minimal and focused: the codebase is a single-script project where correctness and clear numeric shapes matter more than API surface.

## Key files
- `reconstruct_3d_from_omni_directional_img.py`: primary module. Contains `camera_param_to_R_t` and `reconstruct_3d_point` functions. Inspect and modify here.
- `README.md`: currently empty ? add user-facing instructions or a quick example if you add new features.

## Big-picture architecture
- Very small, single-script project. All core logic lives in `reconstruct_3d_from_omni_directional_img.py`.
- Data flow: camera extrinsics (R,t) -> 2D image measurements -> triangulation -> 3D points.
- No service boundaries, no packaging, no external processes.

## Project-specific conventions
- Camera parameter layout (discoverable in the code): expected as 4x3 per camera where
  - rows 0..2 = rotation matrix `R` (3x3)
  - row 3 = translation vector `t` (1x3)
- Functions should accept either a single camera `(4,3)` or batches `(N,4,3)` where practical.
- Numeric arrays use `numpy`; prefer `dtype=np.float64` for stability in linear algebra.

## Recommended implementation patterns for AI edits
- Prefer pure `numpy` vectorized solutions over explicit Python `for` loops when manipulating many cameras or stacking per-camera matrices.
- For triangulation, follow the standard linear method: build per-view projection matrices `P = [R | t]` (shape `(3,4)`), stack the 2M rows of constraints into `A` and solve `AX = 0` with SVD (last right-singular vector). This is efficient and numerically stable for small M.
- Example (adapt into `reconstruct_3d_point`):

```python
# cam: (M,4,3) -> R: (M,3,3), t: (M,3,1)
# P: (M,3,4) formed by np.concatenate([R, t], axis=2)
# pts2: (M,2) -> u,v arrays
# Build A (2M,4): rows u_i*P_i[2]-P_i[0] and v_i*P_i[2]-P_i[1]
# Solve with np.linalg.svd(A) and normalize homogeneous X
```

## Tests / running
- No test harness currently present. To run quick checks or ad-hoc demos, run:

```bash
python reconstruct_3d_from_omni_directional_img.py
```

Add small runnable examples into `__main__` when adding new APIs so humans and bots can validate behavior quickly.

## Integration points & dependencies
- Only dependency is `numpy`. If you add dependencies, update a short `requirements.txt` and document install steps in `README.md`.

## What to avoid
- Do not change public function names without updating the demo or README.
- Avoid guessing camera conventions ? prefer to make code accept either convention explicitly or add a small converter like `camera_param_to_R_t(camera_param)` and document its expected input.

## When you need clarification
- If camera extrinsic format or intrinsic calibration details are unclear, add a short CLI test or unit that converts and prints shapes to confirm assumptions before larger refactors.

---
Please review and tell me any missing details or preferred coding patterns you follow (naming, logging, preferred dtype). I will iterate the instructions accordingly.
