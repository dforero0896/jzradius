# Installation

`jzradius` is a Python package plus two plain C-ABI shared libraries that are
registered with JAX as typed XLA-FFI targets and loaded through `ctypes`. There
is no CPython extension module and nothing links `libpython`, so a wheel is
`py3-none-<plat>`: Python-3.11+ and ABI independent, but platform specific.

Everything below is what the packaging actually does today; nothing is claimed
about PyPI, which does not host a `jzradius` release yet.

## Requirements

| item | needed for | notes |
|---|---|---|
| Python >= 3.11 | always | `pyproject.toml` `requires-python` |
| `jax`, `numpy` | always | installed as dependencies |
| a C++17 compiler (`g++`) | the CPU FFI library | mirrors the CUDA arithmetic, and is the test harness |
| `nvcc` + CUDA toolkit | the CUDA FFI library | only needed for device execution |
| `jztree` (`jztree-cu12`/`jztree-cu13`) | GPU tree construction | optional; without it the GPU path falls back to the median-split scaffold, and tests using the real hierarchy skip |
| JAX's CUDA major == jztree backend major | GPU tree construction | otherwise jztree refuses to import (D002); `JZTREE_SKIP_JAX_CUDA_CHECK=1` bypasses the check |

The build hook finds the XLA FFI headers through `jax.ffi.include_dir()`. Set
`JAXLIB_INCLUDE=/path/to/jaxlib/include` when `jax` cannot be imported into the
build environment (or when building for another Python).

## Install from a source checkout

```bash
git clone git@github.com:dforero0896/jzradius.git
cd jzradius
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e .     # compiles the CPU library, CPU-only
```

`pip install .` (non-editable) behaves the same and additionally stages the
artifacts into the wheel's `jzradius/_libs/`. Both run
`scripts/build_ext.py` — the same driver used by the tests and the documented
workflow — so there is exactly one build recipe.

Set `JZRADIUS_BUILD_CUDA=1` to also compile the device library:

```bash
JZRADIUS_BUILD_CUDA=1 .venv/bin/python -m pip install -e .
# optional: a single arch, instead of the sm_80/86/89/90 default
JZRADIUS_CUDA_ARCH=sm_90 JZRADIUS_BUILD_CUDA=1 .venv/bin/python -m pip install .
```

## Install from an sdist

```bash
pip install jzradius-*.tar.gz        # compiles the CPU library at build time
JZRADIUS_SKIP_NATIVE=1 pip install jzradius-*.tar.gz   # Python only
```

The sdist contains `src/**`, `scripts/**`, `tests/**` and `docs/**` but no
libraries: a checkout's developer-built `jzradius/_libs/*.so` (which may include
a CUDA library for whatever toolkit the login node had) is pruned by
`MANIFEST.in`. This is deliberate — a release artifact must only contain what
its own build produced, which is why `setup.py` compiles into a staging
directory and copies only the fresh files into the wheel.

## Install without native libraries

```bash
JZRADIUS_SKIP_NATIVE=1 pip install .
```

`import jzradius` works and the pure-JAX parts are fully usable
(`reference_histogram`, `math`, `bounds`, `traverse_tree_pairs`,
`build_naive_radius_tree`); calling the kernel-backed public function raises
`FfiUnavailable` with the path it looked for and how to build it. This mode is
also what happens automatically when no compiler or XLA headers are found — the
install prints a warning instead of failing, since the Python layer is useful on
its own.

## Verify an install

```python
import jax, jzradius
from jzradius import ffi

print(jzradius.__version__, jzradius.__file__)
print("backend:", jax.default_backend(), "| ffi platform:", ffi.active_platform())
print("ffi available:", ffi.ffi_available())
```

A quick numerical check that exercises the kernels end to end:

```python
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jzradius

rng = np.random.default_rng(7)
pos = rng.uniform(0.45, 0.55, size=(300, 3))
w = rng.uniform(0.5, 1.5, size=300)
centers = np.linspace(0.005, 0.045, 9)
kw = dict(temperature=0.01, boxsize=1.0, r_max=0.05, pair_convention="unique")

h = jzradius.radius_softmax_histogram(
    pos_query=pos, bin_centers=centers, weights_query=w, self_query=True, **kw)
href = jzradius.reference_histogram(
    pos, pos, centers, weights_query=w, weights_source=np.ones_like(w),
    self_query=True, **kw)
print("relative difference:", np.max(np.abs(h - href) / href))   # ~1e-15

grad = jax.grad(lambda p: jax.numpy.sum(jzradius.radius_softmax_histogram(
    p, bin_centers=centers, weights_query=w, self_query=True, **kw) ** 2))(pos)
```

`self_query=True` with only `weights_query` means `a_ij = w_i` (the source
weight is one); pass `weights_source` for `a_ij = w_i * w_j`, or supply the same
array to both (D006 in `docs/discrepancies.md`). `reference_histogram` defaults
to `self_query=False` and then counts ordered pairs, including `i == i`.

## Known packaging gaps

- No PyPI release and no CI wheel matrix: the CUDA artifact needs a toolkit
  matching the target JAX and an arch list, and this project has only been built
  on one cluster (`docs/environment_lock.md`). Wheels are built on the machine
  that will run them.
- `linux_x86_64` only in practice; nothing in the code is x86-specific, but
  `scripts/build_ext.py` hardcodes the `g++`/`nvcc` invocation style.
- The wheel is deliberately not pure (`py3-none-<plat>`): installing it on an
  incompatible platform would otherwise silently hide the missing library.
- `jztree` is not vendored and not a hard dependency; the CUDA extras only pull
  the matching backend wheel.
