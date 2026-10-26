# Environment Lock

Recorded: 2026-10-24 (Europe/Berlin). Update whenever the interpreter, CUDA
stack, or upstream pin changes.

## Upstream pin

- `jztree` remote: https://github.com/jstuecker/jztree.git
- commit: `39eea8ec22f97551476c5196a94115f4c3a49a1a` (2026-10-16,
  "Use local tree construction on one-device shard maps")
- install mode: **editable** from `/home/dforero/coder/jzradius_dev/jztree`
  (scikit-build-core dev build `1.1.1.dev2+g39eea8ec2`). The clone at the
  pinned commit is what is imported, per the user's instruction; it may be
  edited if upstream fixes are needed.
- backend ABI: the editable finder (`_editable_skbc_jztree.pth`) resolves
  `jztree_cuda` to the clone's `build/` directory, compiled with
  `CUDA_MAJOR = 12`. It is loaded under the CUDA 13 JAX stack by setting
  `JZTREE_SKIP_JAX_CUDA_CHECK=1` (Deviation D004). Runtime validity of that
  mix is proven or refuted at M5 by the trivial FFI op.
- `jztree-cu13==1.1.0` is also installed in `.venv` (27.1 MB wheel) as a
  fallback, but the editable finder takes precedence; it is not currently the
  imported backend.

## JAX / Python

| Component | Version |
|---|---|
| Python | 3.11.17 (`/home/dforero/miniconda3/envs/webjax/bin/python`) |
| jax | 0.10.2 |
| jaxlib | 0.10.2 |
| jax-cuda13-pjrt / jax-cuda13-plugin | 0.10.2 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| pytest | 9.1.1 |

`jax.lib.xla_extension` no longer exists in JAX 0.10.2
(`hasattr(jax.lib, "xla_extension") == False`); see Deviation D001.

## CUDA / toolchain

| Item | Value |
|---|---|
| system toolkit | `/usr/local/cuda` -> CUDA 12.3, `nvcc V12.3.107` |
| pip toolkit | `nvidia-cuda-nvcc==13.4.92`, `nvidia-cuda-runtime==13.4.92`, CCCL 13.3.4 |
| driver (GPU node gn02) | 610.57.04 |
| cmake | 3.26.5 (login node) |
| ninja | not installed on login node; install into `.venv` for CUDA builds |

## Hardware

- Login node `nyx`: no GPU (`nvidia-smi` absent).
- SLURM `gpu` partition (2026-10-24): `gn01 l40s:3 idle`,
  `gn02 a30:2,a40:1 idle`, `gn03 h100:1 idle`, `gn04 a40:3 mix`,
  `gn05 l40s:1 drain`.
- Allocation used so far: gn02, NVIDIA A30 24576 MiB, driver 610.57.04.
- Compute capability: H100 9.0, L40S 8.9, A40 8.6, A30 8.0 (NVIDIA specs;
  re-confirm on-node via JAX device query when GPU tests first run).

## Known platform issue (GPU nodes)

On gn02, a conda-env Python launched through `srun` needed ~18 s just to
print (`real 17.9 s`, `user 0.02 s`), and `import numpy` did not finish
within 120 s at near-zero CPU. `/home` is NFSv3 (`nyx-data100:/home`).
Directory listing is fast, so this is interpreter/site-packages loading, not
GPU or arithmetic. GPU test execution is therefore deferred until a
node-local/containerized interpreter is available; see
`docs/discrepancies.md`. CPU tiers are unaffected (login node).

## Reproduction

```
module load cuda/12.3           # system nvcc for CMake builds
JAX_PLATFORMS=cpu .venv/bin/python scripts/check_env.py --write docs/environment_probe.txt
```

Full dependency snapshot: `docs/pip-freeze.txt` (120 pinned entries).
