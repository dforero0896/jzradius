# Benchmarks

## What has been measured

Both backends. The GPU node stall reported as D003 has since cleared, so the
suite and a timing sweep now run on a real device.

### CUDA kernels on one A30 (24 GB, driver 610.57.04)

`scratch/gpu_bench.py`, float64, `L=1.0`, `r_max=0.05`, 20 bin centers,
`temperature=0.05`, uniform points, `self_query=True`, softmax. Wall clock
includes the host-side tree build and work-item generation (they dominate —
see D012); "cold" is the first call including compilation.

| N | fwd | fwd (cold) | fwd+bwd (`jax.grad`) | ratio bwd/fwd |
|---|---|---|---|---|
| 10 000 | 0.76 s | 10.4 s | 0.87 s | 1.14 |
| 50 000 | 12.4 s | 17.7 s | 12.3 s | 0.99 |
| 100 000 | 3.05 s | 11.3 s | 3.02 s | 0.99 |
| 300 000 | 18.2 s | 25.3 s | 18.4 s | 1.01 |

The `grad` call costs essentially the same as the forward call because the
kernels are a small part of the total: the Python/NumPy tree build and the
work-item descent dominate. The `n=50 000` row is slower than `n=100 000`
because the host build is not monotone in wall clock on this filesystem; treat
these as order-of-magnitude numbers, not a scaling law.

Correctness at the same sizes was checked separately against
`reference_histogram` (values and all four gradients, float64 ~1e-15 relative,
float32 ~1e-6).

### CPU mirror, single-threaded

Host: login node `nyx`, AMD EPYC 7763, Python 3.11.17, JAX 0.10.2
(`JAX_PLATFORMS=cpu`), float64, `leaf_size=32`, `r_max=0.25`, `L=1.0`,
16 bin centers, `temperature=0.05`, uniform points, softmax mode.
Timings are wall-clock including host-side tree construction and work-item
generation; the "steady" column is a second identical call (kernels already
registered/compiled), and `fwd+bwd` is one `jax.grad` of the summed histogram
including retracing.

| N | first call | steady call | fwd+bwd |
|---|---|---|---|
| 512 | 0.152 s | 0.010 s | 0.389 s |
| 2048 | 0.224 s | 0.097 s | 0.358 s |
| 8192 | 1.381 s | 1.219 s | 2.328 s |

These numbers are dominated by the scalar CPU loops and host tree build. They
are useful only as a correctness-path sanity check and to bound the work-item
count; they say nothing about GPU throughput.

## M9 component breakdown (post device work queue and tree balancing)

`scripts/bench_components.py`, median of 5 repetitions, node-local scratch
(`/tmp`, never NFS -- see D003), one A30, float64, `L=1.0`, `r_max=0.05`,
20 bins, `temperature=0.05`, `leaf_size=32`, `self_query=True`, uniform points.
Frozen v1 numbers are in `benchmarks/baseline_v1.json`, the current tree in
`benchmarks/components.json`. `host_frac` = (`wall_fwd` - device stages) /
`wall_fwd`, where the device stages are the jztree build, the device work-item
queue and the histogram kernel.

| N | nodes | work items | jztree build | flatten | host descent | device queue | fwd kernel | wall fwd | wall grad | host frac |
|---|-------|-----------|--------------|---------|--------------|--------------|------------|----------|-----------|-----------|
| 1e4 | 636 | 6 487 | 130 ms | 8 ms | 201 ms | 8.5 ms | 0.7 ms | 0.162 s | 0.164 s | 14.0% |
| 5e4 | 2 885 | 29 761 | 146 ms | 13 ms | 1 021 ms | 9.6 ms | 1.5 ms | 0.180 s | 0.194 s | 12.5% |
| 1e5 | 5 398 | 63 854 | 159 ms | 21 ms | 1 303 ms | 9.6 ms | 2.7 ms | 0.310 s | 0.322 s | 44.7% |

Reading of the table:

- The **device work queue** (D013) is 20-130x faster than the interpreted host
  descent it replaces: 1.30 s -> 9.6 ms at N=1e5. Its item set is compared
  against the host descent on every benchmark case
  (`work_items_device_matches_host`).
- **Tree balancing** (D014) removed the flat-hierarchy cliff: jztree returns a
  root-with-all-leaves tree below ~6e4 particles, so N=5e4 used to cost
  2.31e6 node-pair visits in the host descent. After grouping the leaves into
  4-ary parents the same case needs 2.17e5 visits and the old N=5e4 anomaly
  (1.26 s, 4-7x slower than N=1e5) is gone: 0.180 s.
- **v1 comparison** (`baseline_v1.json`, same node, same protocol) at N=1e5:
  wall fwd 2.906 s -> 0.310 s (9.4x), wall grad 2.916 s -> 0.322 s, host
  fraction 95.4% -> 44.7%.
- `bwd/fwd` = `wall_grad/wall_fwd` = 1.01-1.08, well inside the <= 2.0 target.
- The remaining host time is the Python call path around the kernels
  (input validation, `jnp.take` permutation, `ffi_call` dispatch, device->host
  sync), not tree construction. Getting it under the plan addendum's 10%
  target at N=1e5 needs the jit-able entry point (M9.2, D008), where that
  Python work is traced away instead of re-executed per call.
- N=1e6 was **not** run: it is a scale-tier measurement that needs approval
  (PLAN.md 2.6), and with 20 neighbours it would need ~1e7 interactions
  (~120 MB) for the interaction list alone.

## Targets from the plan (§21)

- 10^6 particles forward: seconds, not minutes.
- backward: no more than ~3x forward.
- Correctness outranks speed at every gate.

## How to reproduce the GPU numbers

```bash
module load cuda/12.3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
srun -p gpu --gres=gpu:1 -n1 -t 1:00:00 \
  .venv/bin/python scripts/run_benchmark.py --n 1000000 --tier scale
```

Record: GPU model and driver, `nvidia-smi` output, N, `r_max/L`, mean
neighbours, leaf size, forward seconds, backward seconds, peak memory, and the
work-item count (`TraversalStats.max_queue_size` / `n_leaf_pairs`). Scale tier
(N >= 10^6) requires explicit approval per PLAN §2.5.
