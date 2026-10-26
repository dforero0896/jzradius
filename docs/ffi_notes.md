# FFI and build notes

## Chosen path

Typed XLA FFI, `api_version=1`, registered per platform through
`jax.ffi.register_ffi_target`. This is the same mechanism `jztree` uses
(`jax.ffi.register_ffi_target("PosZorderSort", ...)` then `jax.ffi.ffi_call`),
so there is one registration style in the process and no legacy
`xla_client` custom-call plumbing.

Registration (`jzradius/ffi.py`):

```python
lib = ctypes.CDLL(path)                     # RTLD_LOCAL by default
jax.ffi.register_ffi_target(
    "jzradius_forward",
    jax.ffi.pycapsule(getattr(lib, "JzRadiusForward")),
    platform="cpu" or "CUDA",
    api_version=1,
)
```

The PyCapsule name XLA looks for is `xla_ffi_function_v1`; `jax.ffi.pycapsule`
creates it. This was verified end-to-end with a trivial `y = x + 1` handler
before any radius code was written (M5 gate).

Calls go through `jax.ffi.ffi_call(target, out_shapes, vmap_method="sequential")`
with `custom_call_api_version` left at the current default (4). Scalar
parameters are passed as FFI **attributes**, not buffers, so no host-device
copies occur for them:

| attribute | type | meaning |
|---|---|---|
| `self_query`, `unique_pairs`, `mode` | `int32` | pair-convention switches |
| `r_max`, `boxsize`, `temperature`, `eps`, `r_safe`, `pair_scale` | `double` | numerics |

Both kernels take `Ctx<PlatformStream<cudaStream_t>>` (CUDA) or the CPU stream
context, so launches run on the XLA-provided stream; no secondary stream is
created, which keeps ordering with surrounding JAX work.

## Output buffers

XLA does **not** guarantee zero-initialized output buffers for
`Result<AnyBuffer>`. Both handlers therefore clear their outputs themselves:
`cudaMemsetAsync` on the CUDA side (before the early-out for `n_work == 0`,
otherwise an empty interaction list would return garbage) and `std::memset` in
the CPU mirror. This was found the hard way: the first end-to-end run produced
~1e243 garbage before the memset was added.

## Layout of the code

```
src/cuda/jzradius_kernels.cuh   shared arithmetic (host & device), no CUDA-only types
src/cuda/jzradius_cuda.cu       __global__ kernels + typed-FFI handlers (nvcc)
src/cpu/jzradius_cpu.cpp        CPU mirror using the same header (g++)
jzradius/ffi.py                 ctypes loading, registration, attrs, calls
jzradius/histogram.py           custom_vjp wrapper around forward/backward
scripts/build_ext.py            incremental build of both artifacts
```

The shared header is the single source of truth for MIC distance,
softmax/Gaussian weights, the `A/B/C` streaming statistics and the periodic
box lower bound. The CPU artifact is therefore a behavioural mirror of the GPU
kernels, which is what makes the CPU test suite meaningful while the cluster's
GPU nodes are unusable: the tests exercise the same math, the same FFI
registration and the same `custom_vjp` path, only with a different backend.

## Exported symbols

Only these three have default visibility (built with
`-fvisibility=default`; `hidden` silently produced `undefined symbol` at
`ctypes` load time):

- `JzRadiusForward` / `JzRadiusForwardCpu`
- `JzRadiusBackward` / `JzRadiusBackwardCpu`
- `JzRadiusNodeBounds` / `JzRadiusNodeBoundsCpu`

## Build

```bash
python scripts/build_ext.py            # honour source mtimes
python scripts/build_ext.py --force
python scripts/build_ext.py --no-cuda
```

- CPU: `g++ -O3 -std=c++17 -shared -fPIC -fvisibility=default` with
  `jax.ffi.include_dir()` on the include path.
- CUDA: `nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC
  -Xcompiler -fvisibility=default --expt-relaxed-constexpr`, one `-gencode`
  per architecture. Default set is `sm_80, sm_86, sm_89, sm_90` covering the
  cluster's A30/A40, L40S and H100; override with
  `JZRADIUS_CUDA_ARCH=sm_90` (comma-separated).
- If `nvcc` is not on `PATH` the script also checks `/usr/local/cuda*/bin`;
  on this cluster the login node needs `module load cuda/12.3`.
- Artifacts land in `jzradius/_libs/` and are loaded lazily; a missing
  artifact raises `FfiUnavailable` with the build command in the message.

## Why `custom_vjp` (and not `jax.custom_jit` / `pure_callback`)

`ffi_call` is opaque to autodiff, and tree construction is host-side (jztree
runs its own FFI kernels and allocates host arrays). The chosen split is:

- forward `custom_vjp` rule: validate + build tree + `build_work_items` +
  `ffi.forward`, returning the histogram and a residual tuple;
- backward rule: `ffi.backward` with the incoming histogram cotangent, then
  scatter-add back to the original index order.

Consequence: the public call must be **eager** in v1. The tree build needs
concrete host data, so `jax.jit(jax.grad(...))` around the public function
raises `JzRadiusError` with an explanatory message rather than silently
mis-tracing. `jax.grad` / `jax.value_and_grad` work eagerly, which covers the
intended use. This is recorded as deviation D008.

## Determinism

Both kernels accumulate with atomics. `deterministic=True` is rejected (D007)
instead of being accepted and quietly producing run-to-run variation. A
deterministic path would need sorted or per-block reduction accumulation.
