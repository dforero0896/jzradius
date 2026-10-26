# Upstream API Notes — jztree @ 39eea8e

Static reading of the clone (editable install in `webjax`, so HEAD == installed).
Paths are relative to the `jztree` repo root. Unread/unavailable facts are UNKNOWN;
no GPU code was run.

## 1. Tree construction / geometry

```python
def zsort(pos: jax.Array | Pos) -> Tuple[jax.Array | Pos, jax.Array]            # tree.py:96
def zsort_and_tree(part: Pos, cfg_tree: TreeConfig = TreeConfig(),               # tree.py:377
                   data: Any | None = None, ptype: jax.Array | None = None,
                   num_types: int | None = None, shrink: bool = True) -> Tuple[Pos, TreeHierarchy]
def build_tree_hierarchy(partz: PosMass | jax.Array, cfg_tree: TreeConfig,       # tree.py:781
                         lvl_bound: jax.Array | None = None, ptype: jax.Array | None = None,
                         num_types: jax.Array | None = None) -> TreeHierarchy
def detect_leaf_boundaries(posz: jax.Array, ptype: jax.Array | None = None,      # tree.py:191
        num_types: int | None = None, leaf_size: int = 32, npart: int | None = None,
        lvl_bound=None, block_size: int = 64, alloc_size: int | None = None,
        cfg_reg: RegularizationConfig | None = None) -> jax.Array
def determine_znode_boundaries(posz: jax.Array, block_size: int = 64,            # tree.py:270
        nleaves: jnp.array = None) -> Tuple[jax.Array, jax.Array, jax.Array]
def get_node_geometry(posz: jax.Array, lbound: jax.Array, rbound: jax.Array,     # tree.py:292
        num: jax.Array | None = None, block_size: int = 64,
        result: str = "lvl_cent_ext", upper_extent: bool = False) -> Tuple[jax.Array, jax.Array, jax.Array]
```

- `zsort` returns `(posz, idz)` with `posz = pos[idz]`; inverse is
  `jztree.tools.inverse_indices(idz)` (`tools.py:127`). It is `custom_vjp`-wrapped
  (`tree.py:112-139`) around FFI target `"PosZorderSort"` (`tree.py:88`).
- `zsort_and_tree` returns `(partz, th)` plus optional data/ptype entries when
  supplied (`tree.py:426-432`); single-device path is `zsort` +
  `build_tree_hierarchy`.
- Plane 0 = leaves; coarsest plane `num_planes()-1` = root (`tree.py:790-798`,
  `fof.py:146`). `build_tree_hierarchy` needs already z-sorted input.
- `detect_leaf_boundaries` returns `splits` of length `N+1`, `splits[0]=0`,
  `splits[-1]=npart`; no type may exceed `leaf_size` particles per leaf
  (`tree.py:202`).
- `determine_znode_boundaries` returns `(lvl, lbound, rbound)`; in
  `_define_split_hierarchy` it is called on `posz[ispl[:-1]]` with
  `nleaves=argmax(ispl)` (`tree.py:680-681`). `lvl` is a bit-interleaved Morton
  level; `lbound`/`rbound` are inclusive/exclusive particle indices.
- `get_node_geometry` `result` is an underscore-joined subset of
  `{"lvl","cent","ext"}`; multi-key output follows that order (`tree.py:332-339`).
  Bounds are `pos[lbound]`/`pos[rbound-1]` and the node is the smallest common
  Morton cell (`tree.cuh:327-348`); invalid level is `-2000`.
  **`ext` is full width, not half-width** (`test_tree.py:46-47`,
  `LvlToExt` `math.cuh:516`): box = `cent ± ext/2`. `upper_extent=True` uses
  parent-level-1 cells and may include empty space (`tree.cuh:350-364`).

## 2. `TreeHierarchy` / `PackedArray`

`PackedArray` (`data.py:253`): fields `data`, `ispl` (len `nlevels+1`),
`fill_values`, `levels_filled`; methods `get(level,size,fill_value)`,
`set`, `append`, `size()`, `num(level)=ispl[l+1]-ispl[l]` (`:333`),
`nfilled()` (`:336`), `nlevels()` (`:339`).

`TreeHierarchy` (`data.py:388-466`): `size_leaves:int` (static), `ispl_n2n`,
`ispl_n2l`, `ispl_l2p_per_type:List[jax.Array]`, `lvl`, `geom_cent`, optional
`mass`/`mass_cent`.

| Field | Meaning |
|---|---|
| `ispl_n2n` | node→child-node splits per plane; `get(0)` = leaf→particle splits |
| `ispl_n2l` | node→leaf index, one entry per node |
| `ispl_l2p_per_type` | leaf→particle splits, one array per type |
| `lvl` | Morton level per node (`PackedArray`) |
| `geom_cent` | geometric center `(M,3)` per node (`PackedArray`) |

Methods: `num(level)=lvl.num(level)` (`:453`), `num_planes()` (`:449`),
`size()` (`:457`), `npart(level,ptype=0)` (`:426`), `splits_leaf_to_part`
(`:419`), `center()` = mass_cent if present else geom_cent (`:433`).
**No `node_left/right/parent/lo/hi` fields exist**; this is a hierarchy of planes
with variable branching, not a binary tree. Boxes are derived from center + level.

Node particle range: `leaf0 = ispl_n2l.get(level)[i]`,
`lo = ispl_l2p_per_type[0][leaf0]`,
`hi = ispl_l2p_per_type[0][ispl_n2l.get(level)[i+1]]`. Out-of-range split
lookups return a fill value (next finer plane's count) (`tree.py:755-757`).

## 3. `InteractionList`

`data.py:474-528`:
```python
@dataclass(slots=True)
class InteractionList:
    ispl: jax.Array                     # (n_query_nodes+1,) int32
    isrc: jax.Array                     # (n_entries,) int32 source-node indices
    rad2: jax.Array | None = None       # per-entry radius^2, kNN only
    ids: jax.Array | None = None        # multi-GPU origin ids
    dev_spl: jax.Array | None = None    # multi-GPU rank splits
    has_separate_query_and_source_indices: bool = False
```
Query node `i` interacts with `isrc[ispl[i]:ispl[i+1]]` (`:477-479`).
`nfilled()=ispl[-1]`; `size()=isrc.size` (`:518-524`). Entries are **node
indices**, not particle ranges; leaf kernels get a separate leaf-split array.
Builders: `grouped_dense_interaction_list` (`tree.py:898`),
`distr_grouped_dense_interaction_list` (`tree.py:967`),
`simplify_interaction_list` (`tree.py:1019`).
## 4. FoF traversal and periodic bounds

```python
def _fof_dual_walk(th: TreeHierarchy, rlink: float, boxsize: float = 0.,         # fof.py:125
        alloc_fac_ilist: int = 128) -> Tuple[FofNodeData, InteractionList]
def _fof_node2node(ilist: InteractionList, spl_parent: jax.Array,               # fof.py:32
        node_data: PosLvl, node_igroup: jax.Array, rlink: float,
        boxsize: float = 0., block_size: int = 32) -> Tuple[jax.Array, InteractionList]
def _fof_leaf2leaf(leaf_data: FofNodeData, ilist: InteractionList,              # fof.py:167
        posz: jax.Array, rlink: float, boxsize: float = 0., block_size=32)
def fof_labels(part: jax.Array | Pos, rlink: float, boxsize: float = 0.,        # fof.py:816
        cfg: FofConfig = FofConfig(), th: TreeHierarchy | None = None,
        output_order: str = "z") -> Tuple[jax.Array | Pos, jax.Array]
```
- `_fof_dual_walk` is **self-interaction only**, top plane→leaves via
  `lax.fori_loop` (`:144-160`); `FofNodeData=(lvl,igroup,spl)` (`data.py:592`).
  `_fof_leaf2leaf` returns union-find labels, not pairs.
- `fof_labels` with `th` assumes particles already z-sorted (`:830-831`); `output_order` is `"z"` or `"input"`.
- CUDA bounds (`fof.cuh:158-159`) pass `nodeQ.extent + lT.extent` as `width_half`;
  nodes are built with `NodeLvlToHalfExt` (`fof.cuh:106,133`, `math.cuh:529`), so
  these are **half-widths**. `:164` links when
  `max(r2max, 4*|extent|^2) <= r2link`; `:175` appends when `r2min <= r2link`.
- `math.cuh`: `wrap_dx` (`:417`) wraps into `[-L/2,L/2)`, no-op for
  `boxsize<=0`; `distance_squared` (`:457`); `mindist2(center1, center2,
  width_half, boxsize=0)` (`:574`) = `Σ max(|wrap(dcent_i)|-width_half_i,0)^2`;
  `maxdist2` (`:592`) = `Σ (|wrap(dcent_i)|+width_half_i)^2`. `mindist2` is a
  valid lower bound, `maxdist2` a loose upper bound, for scalar cubic boxes; tree
  construction itself is not periodic-aware (only distance kernels are).
## 5. Radius query: absent

No standalone radius/cutoff/pair-count API exists in `src/jztree/*.py` (grep for
`radius|cutoff|r_max|r_search|within` finds only `com_inertia_radius` in the FoF
catalogue). Closest are **FoF** (`fof.py:816`: node list at fixed `rlink`, reduced to
connected-component labels, never emits/counts pairs) and **kNN** (`knn.py:205`:
fixed integer `k`; `InteractionList.rad2` holds kth-neighbour pruning radii, not
a user radius).

`search_sorted_z` (`tree.py:144`) is a 1-D z-order insertion-point lookup, not a
range query. For a radius traversal, reuse the interaction-list pattern
(`grouped_dense_interaction_list` + node2node) and write a leaf-leaf pair kernel;
`_fof_dual_walk` is the best structural template but needs query/source
separation and removal of the `igroup` union-find state.
## 6. Adapter gotchas

- `sorted_to_orig = idz`; `orig_to_sorted = inverse_indices(idz)`. `get_node_geometry` `ext` is full width; `mindist2` wants half-width.
- Only centers+levels are stored; boxes = `cent ± 0.5*2**lvl` (conservative Morton cells).
- Plane 0 = leaves, `num_planes()-1` = root; no binary left/right/parent. Materialize if `RadiusTree` requires it (PLAN §9.4 option 2).
- Arrays are padded to an allocation; use `th.num(level)`/`nfilled()` and guard
  leaf ranges with `ispl_l2p_per_type`. `boxsize` is one float for all dimensions;
  `0.` disables wrapping. No Python MIC helper is exposed outside CUDA.
