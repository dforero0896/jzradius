# Algorithm

Scope: exact periodic radius pair counting with smooth (softmax or Gaussian)
bin weights, differentiable in positions and in per-particle weights.
This document describes the implemented path, not the aspirational one.

## 1. Tree adaptation (`jzradius/tree_adapter.py`)

`jztree` stores a Morton (z-order) hierarchy as *level planes*: for each level
`l`, one entry per node, plus two prefix-sum ("split") arrays `ispl_n2l`
(node → covering leaves) and `ispl_n2n` (node → children). The kernel wants a
flat node array, so `build_radius_tree_from_levels` flattens the planes:

- level 0 = finest (leaf) level; level `NL-1` = coarsest;
- nodes are numbered level by level, children before parents, so a child id is
  always strictly smaller than its parent id;
- a synthetic root covering the whole periodic box is appended and made the
  parent of the coarsest real node;
- `node_start`/`node_count` are particle ranges, `node_left`/`node_right` are
  the contiguous child range (a partition of the child id space, because ids
  are assigned contiguously per parent), `node_level`, `node_parent`,
  `node_lo`/`node_hi`/`node_center`/`node_radius` complete the record.

Boxes: a leaf box is the exact particle min/max in that leaf; an inner box is
the union of its children's boxes; `node_radius` is half the box diagonal.
This is conservative by construction (PLAN §9.5 priority 3: tight particle
bounds, not Morton-cell bounds) and is validated by `check_tree_invariants`:
invertible `sorted_to_orig`/`orig_to_sorted`, disjoint complete leaf
partitioning, child-count consistency, box containment, `root.count == N`.

`build_naive_radius_tree` implements the same flat layout with median splits
for CPU tests; it is a test scaffold, never the production tree.

## 2. Periodic lower bounds (`jzradius/bounds.py`)

For two axis-aligned boxes and scalar box length `L`, the minimum-image gap on
one axis is

```
gap_1d(a0,a1,b0,b1) = min over s in {-L,0,+L} of max(0, b0+s - a1, a0 - (b1+s))
```

and the 3-D bound is `sqrt(gap_x^2 + gap_y^2 + gap_z^2)` (PLAN §10.3). It is
provably a lower bound on the true minimum-image particle distance, so
`lb > r_max + tol` is a safe prune. The tolerance is `8 * eps * L`
(`prune_tolerance`). A bounding-sphere variant (`lower_bound_spheres`) is
available for callers that prefer tighter spheres.

## 3. Dual-tree traversal and work-item generation

`jzradius/traversal.py::traverse_tree_pairs` is the CPU reference: an explicit
stack of `(node_q, node_s)` pairs. The larger side is split (or the only
splittable side), a node pair is pushed back when its lower bound is within
`r_max + tol`, and two leaves enumerate particle pairs.

`build_work_items` performs the same descent but emits only leaf pairs as
`int32 (q0,q1,s0,s1)` rectangles (exclusive ends), which is the unit of work
for both the forward and the backward target. Properties:

- self-query: `q0..q1` and `s0..s1` are the same leaf; the rectangle is walked
  as an upper triangle (`unique`) or fully ordered (`directed_half`);
- cross: two independent trees/leaf ranges;
- overflow is explicit: more than `max_work_items` items raises
  `WorkQueueOverflow`; nothing is silently dropped;
- work items depend only on geometry and `r_max`, so the backward pass reuses
  the forward's list instead of re-traversing with different rounding.

## 4. Pair conventions

| setting | enumerated | scale |
|---|---|---|
| `self_query=True, "unique"` | unordered `i < j` once | 1 |
| `self_query=True, "directed_half"` | ordered `i != j` | 1/2 in the differentiable graph |
| cross-correlation (both args) | full cartesian product | 1 |

For `self_query=True, unique`, "i < j" is defined in **input index space**
(the traversal visits tree-sorted indices and maps back before reporting).
See D006 for the consequence with asymmetric weights.

## 5. Distances and bin weights

Single minimum image per axis, then

```
d_k = (x_q,k - x_s,k) - L * round((x_q,k - x_s,k)/L)
r   = sqrt(sum_k d_k^2 + eps)          eps = 1e-60 (f64) / 1e-30 (f32)
```

`round` is treated as a constant (its derivative is zero a.e.), so
`dr/dx_q,k = d_k / r`. Pairs with `r > r_max` are skipped; the backward pass
additionally skips `r < r_safe = max(1e-12 L, eps)` to avoid `1/r` blow-up.

Softmax weights, with `alpha = 1/T^2`, `q_b(r) = -alpha (r - c_b)^2`:

```
m      = max_b q_b
W_b    = exp(q_b - m) / sum_c exp(q_c - m)
q'_b   = -2 alpha (r - c_b)
```

Gaussian mode uses **unnormalized** `W_b = exp(q_b)` (PLAN §6.4). The
max-shift is applied only in softmax mode, where it is needed for stability;
for Gaussian `q_b <= 0`, so `exp(q_b)` cannot overflow.

Histogram: `H_b += a_ij * W_b`, with `a_ij = wq[i] * ws[j] * pair_scale`.

## 6. VJP derivation (`custom_vjp`)

Let `g_b = dL/dH_b` and define, per accepted pair,

```
A = sum_b g_b W_b
B = sum_b W_b q'_b
C = sum_b g_b W_b q'_b
```

Then

```
softmax : dL/dr = a_ij * (C - A B)      (the -A B term is the softmax Jacobian)
gaussian: dL/dr = a_ij * C
s       = dL/dr / r
grad x_q +=  s * d
grad x_s += -s * d
grad wq[i] += pair_scale * ws[j] * A
grad ws[j] += pair_scale * wq[i] * A
```

The kernels compute `A, B, C` in one streaming pass over bins (no per-bin
scratch, so `n_bins` is unbounded); the forward does a second pass to write
`H`. Accumulation into particles and weights uses `atomicAdd` on GPU and plain
`+=` in the CPU mirror.

`custom_vjp` is used because `ffi_call` is not differentiable. The forward
returns the histogram and closes over the residual
`(perm_q, perm_s, centers, work, params, sorted positions/weights)`; the
backward runs the VJP target and scatters gradients back with
`jnp.zeros(...).at[perm].add(...)`. For self-query, forward and backward share
the same position array, so the backward's query and source gradients are
summed along with the two weight gradients (`fold_ws_into_wq`), avoiding
double-counting of the same residual in the outer `jax.grad`.

## 7. Complexity

With `N` particles, leaf size `l`, and `P` accepted pairs, the traversal is
`O(N log N)` for tree construction plus `O(P)` for the pair work; pruning makes
the constant depend on density and `r_max`, not on `N^2`. Work-item generation
is `O(number of leaf pairs within range)`.
