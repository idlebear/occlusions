# GPU Discrete OCE Acceleration Plan

## Summary
Move discrete OCE to the existing PyCUDA stack, preserving CPU `_exact_entropy` semantics within numeric tolerance. The new path should keep all large tensors on device, compute visibility and entropy scores on GPU, and copy back only per-path scores by default. The CPU implementation remains as a correctness fallback and test oracle.

## Key Changes
- Add a GPU discrete OCE backend, likely under `warp_mppi/legacy/`, exposed through a small wrapper such as `evaluate_discrete_oce_gpu(...)`.
- Replace the current CPU path:
  - Python/Shapely `build_path_occlusion_schedule`
  - per-path/per-agent `evaluate_method("discrete_exact_entropy", ...)`
  - dense HMM transition propagation
- Use device-resident inputs:
  - candidate paths as `(num_paths, horizon + 1, 2)`
  - SDD state centers/grid ids
  - sparse CSR transition matrices per destination class
  - per-agent belief vectors and class priors
  - static occupancy grid or static polygon rasterization compatible with existing CUDA visibility code
- Return only:
  - `best_trajectory`
  - per-path entropy scores
  - optional debug tensors only when `--debug-discrete-oce` or equivalent is enabled.

## Implementation Plan
- First add instrumentation around the current discrete path:
  - HMM update time
  - visibility mask construction time
  - `_exact_entropy` time per path/agent
  - host-device transfer time once GPU path exists
- Refactor discrete model storage so SDD transitions remain sparse:
  - stop converting all class transitions to dense arrays for runtime scoring
  - create one shared transition/model object per SDD scene
  - avoid recomputing HMM `l2_differences` / `kl_divergences` per agent when not needed
- Implement GPU HMM update:
  - batch all seen agents in one launch
  - visible agent: collapse/update belief to observed state and update class posterior using emission likelihood
  - unseen-but-known agent: propagate `b_next = sum_c p(c) * b @ P_c`
  - keep agent beliefs and class priors device-resident across ticks
- Implement GPU visibility update:
  - raster/grid-based line-of-sight for every `(path, step, state)` using existing PyCUDA visibility conventions
  - output an occlusion tensor or visibility tensor with shape `(num_paths, horizon + 1, num_states)`
  - treat `1 = occluded`, `0 = visible` at the entropy boundary to match `_exact_entropy`
- Implement GPU exact entropy:
  - reproduce `_exact_entropy` partition semantics:
    - unseen partition: occluded through step `k`
    - seen-at-state/time partitions: visible at partition step, occluded afterward
  - use sparse CSR transition propagation instead of dense suffix matrices
  - parallelize over `(path, agent, step, partition/state)` and reduce entropy contributions per path
  - accumulate the same score currently used for selection: cumulative entropy over horizon, summed over agents
- Wire CLI behavior:
  - keep `--oce-eval-method trajectory|discrete`
  - add or reuse a backend flag such as `--discrete-oce-backend cpu|gpu`
  - default to GPU when available, fall back to CPU with a warning if PyCUDA/context setup fails.

## Test Plan
- Unit tests with tiny hand-built Markov chains:
  - compare GPU vs CPU `_exact_entropy` for fixed occlusion schedules
  - include all-hidden, all-visible, no-agent, one-agent, and multi-class cases
- Visibility tests:
  - compare GPU visibility masks against the current CPU/Shapely result on small SDD-like grids
  - include static obstacle blocking and scan-range cutoff
- Integration smoke test:
  - run discrete OCE for a few paths and 1-5 agents, assert same best path as CPU within tolerance
- Performance target:
  - 5 visible agents, normal SDD scene, current horizon: state update under 50 ms and OCE scoring under 200 ms after warmup
  - debug tensor copy disabled for timing runs.

## Assumptions
- Use PyCUDA, not Warp, because the repo’s active MPPI/OCE path already uses PyCUDA and the Warp OCE backend is documented as stale.
- Preserve CPU `_exact_entropy` behavior within floating point tolerance; no pruning/top-k approximation in the first GPU version.
- Normal simulation copies back scores only; full visibility/belief tensors are debug-only.
