# Assumption Audit — jztree @ 39eea8e

Audit target: local clone `jztree`, commit `39eea8e` (`1.1.1.dev2+g39eea8ec2`).
The clone is installed editable in the `webjax` conda env, so "installed" ==
HEAD source. Paths below are relative to the `jztree` repo root. All findings
are from static reading; no GPU code was run.

| ID | Assumption | Status | Evidence (file:line) | Finding |
|---|---|---|---|---|
| A1 | Uses Morton / z-order trees, not KD-trees | CONFIRMED | `src/jztree/tree.py:88`, `src/jztree_cuda/sort.cuh:41`, `src/jztree_cuda/common/math.cuh:327`, `src/jztree/data.py:377` | Sorting is by `z_pos_less` bit-order (`PosZorderSort`), and node levels are explicitly "Morton level"; there is no KD-tree code path. |
| A2 | A usable tree / tree-plane hierarchy can be obtained or constructed | CONFIRMED | `src/jztree/tree.py:377`, `src/jztree/tree.py:781`, `src/jztree/data.py:388` | `zsort_and_tree` returns `(partz, th)` and `build_tree_hierarchy` exposes the full `TreeHierarchy`; planes are indexed 0=leaves … `num_planes()-1`=root. |
| A3 | Node particle ranges in Morton-sorted order are accessible | CONFIRMED | `src/jztree/data.py:396-398`, `src/jztree/tree.py:753-757`, `src/jztree/data.py:419-431`, `src/jztree/data.py:453` | `ispl_n2n.get(0)` is the leaf→particle split array; internal node ranges go through `ispl_n2l` + `ispl_l2p_per_type`; `npart(level)` returns counts. `zsort` returns `idz` for the permutation. |
| A4 | Node bounding boxes or Morton cell extents are available | CONFIRMED | `src/jztree/tree.py:292-340`, `src/jztree_cuda/tree.cuh:327-371`, `checks/tests/test_tree.py:46-47`, `src/jztree_cuda/common/math.cuh:516` | `get_node_geometry(..., result="lvl_cent_ext")` returns level, center, extent. **`ext` is the full cell width**, not a half-width; boxes are `center ± ext/2`. `TreeHierarchy` stores only `lvl` and `geom_cent`, not `ext`, so extents must be recomputed from level. |
| A5 | Periodic minimum-image handling exists and is reusable | CONFIRMED | `src/jztree_cuda/common/math.cuh:417-433`, `:457-467`, `:574-589`; `src/jztree/fof.py:34`, `src/jztree/knn.py:225` | `wrap_dx` maps a displacement to `[-L/2, L/2)` and `distance_squared`/`mindist2` take a scalar `boxsize` (0 disables wrapping). Python APIs accept a scalar float `boxsize`; no per-axis lengths or Python MIC helper. |
| A6 | A dual-tree traversal / interaction-list mechanism exists and is reusable | CONFIRMED | `src/jztree/fof.py:125`, `src/jztree/knn.py:138`, `src/jztree/data.py:474-501`, `src/jztree/tree.py:898` | `InteractionList` maps query-node segments (`ispl`) to source node indices (`isrc`), with optional `rad2`; both FoF and kNN have level-by-level dual walks and dense top-level list builders. |
| A7 | An exact radius/cutoff search already exists, or can be added cheaply | **REFUTED** (no existing query) | `src/jztree/knn.py:205-215`, `src/jztree/fof.py:816-818`, `src/jztree/tree.py:144` | There is **no standalone radius, cutoff, or pair-count API** anywhere under `src/jztree`. kNN requires a fixed `k`; FoF requires `rlink` and returns group labels only. The closest primitives are the FoF node→node list + leaf2leaf kernel; adding a radius traversal is feasible by copying that structure, but it does not exist today. |
| A8 | The FoF node-node dual walk can template a pair-count traversal | CONFIRMED (with caveats) | `src/jztree/fof.py:125-165`, `src/jztree_cuda/fof.cuh:82-189`, `src/jztree/fof.py:167` | Structure is reusable: coarse→fine planes, per-parent interaction list, `mindist2`/`maxdist2` pruning, then leaf-leaf resolution. Caveats: the walk is **self-interaction only** (`fof.cuh:121-122` skips `parentT < parentQ`), it threads union-find `igroup` state through every level, and it materializes whole-plane lists instead of a stack. A cross-correlation / generic pair counter must add query-vs-source separation and its own leaf kernel. |
| A9 | Installed JAX FFI supports custom CUDA forwards and backwards | CONFIRMED | external `site-packages/jax/_src/ffi.py:384,644-657` (webjax env, JAX 0.10.2); `src/jztree/tree.py:112-139` | `ffi_call` itself is **not differentiable** (JVP/transpose raise and tell you to use `custom_jvp`/`custom_vjp`). The supported pattern is wrapping the forward FFI call in `jax.custom_vjp`, exactly as `zsort` does; a separate FFI call can be launched inside the backward. |
| A10 | jz-fmm uses analytic radial derivatives + backward kernels, not traced CUDA | UNKNOWN | `README.md:9` only links to jz-fmm | No `jz-fmm` checkout or installed package exists in the workspace (`find`/`glob` found none), so its differentiable implementation could not be inspected. |
| A11 | Documented `knn` / `fof` signatures match the installed version | CONFIRMED | `src/jztree/knn.py:205-215`, `src/jztree/fof.py:816-818`; external dist-info `jztree-1.1.1.dev2+g39eea8ec2` | The editable install points at this clone, and the source signatures are the real installed ones: `knn(part, k, boxsize=None, th=None, part_query=None, result="rad_globalidx", reduce_func=None, output_order="input", cfg=KNNConfig())`; `fof_labels(part, rlink, boxsize=0., cfg=FofConfig(), th=None, output_order="z")`. |
| A12 | jz-fmm boundary conditions are or are not periodic | UNKNOWN | `README.md:9` only | No jz-fmm source/config is available locally; cannot verify. |

## Load-bearing notes

- **A7 is the main plan correction.** PLAN §3.2 words A7 as "already exists or
  can be added cheaply". The first half is false: no radius/cutoff/pair-count
  query exists. The second half is plausible but unproven; budget the adapter
  and CPU traversal as new code, not as a wrapper around an existing query.
- **A8 is a template, not a drop-in.** FoF's walk is the right shape, but the
  `igroup` union-find logic, the "evaluate each pair once" parent ordering, and
  the fixed `r2link` are baked into both the Python and CUDA layers. For a
  cross-correlation `(q, s)` traversal, the query/source distinction must be
  added end to end.
- **A4 box convention is a correctness trap.** `get_node_geometry` returns full
  extent; the CUDA `mindist2`/`maxdist2` helpers expect half-widths, and FoF
  converts via `LvlToHalfExt` (`math.cuh:529`, used at `fof.cuh:106,133`).
  Using `ext` directly as `width_half` would over-prune and silently miss pairs.
- **A9 environment caveat.** The base conda env has no JAX at all; the
  `webjax` env has JAX 0.10.2 and the editable jztree install (CUDA 12 backend
  under CUDA 13 JAX per the parent, unverified here). No GPU code was run.
- **A10/A12 are honest UNKNOWNs.** jz-fmm would have to be cloned/inspected to
  answer them; nothing in this workspace supports a guess.
