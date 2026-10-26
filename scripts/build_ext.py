#!/usr/bin/env python3
"""Build the jzradius FFI shared libraries (PLAN.md 5, M5).

Two artifacts are produced under ``jzradius/_libs``:

* ``libjzradius_cpu.so``  - CPU mirror of the device kernels, built with g++.
* ``libjzradius_cuda.so`` - CUDA kernels, built with nvcc when available.

Both use the typed XLA FFI (``api_version=1``) and the same math header, so the
CPU artifact is an exact behavioural mirror, not a different algorithm.

Usage
-----
    python scripts/build_ext.py [--force] [--no-cuda] [--out DIR]

``JAXLIB_INCLUDE`` points the compiler at the XLA FFI headers when ``jax`` is not
importable in the building interpreter (the pip build hook uses it), ``CXX`` and
``NVCC`` select the compilers, and ``JZRADIUS_CUDA_ARCH`` overrides the default
``sm_80/86/89/90`` fat binary.

The script is idempotent: it rebuilds only when a source file is newer than the
artifact unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "jzradius" / "_libs"

CPU_SOURCES = [
    SRC / "cpu" / "jzradius_cpu.cpp",
    SRC / "cpu" / "jzradius_traverse_cpu.cpp",
]
CUDA_SOURCES = [
    SRC / "cuda" / "jzradius_cuda.cu",
    SRC / "cuda" / "jzradius_traverse.cu",
]
HEADERS = [
    SRC / "cuda" / "jzradius_kernels.cuh",
    SRC / "cuda" / "jzradius_cuda.cu",
    SRC / "cuda" / "jzradius_traverse.cu",
    SRC / "cpu" / "jzradius_cpu.cpp",
    SRC / "cpu" / "jzradius_traverse_cpu.cpp",
]

# GPU architectures the cluster provides (A30/A40 sm_86, L40S sm_89, H100 sm_90).
DEFAULT_ARCHES = ("sm_80", "sm_86", "sm_89", "sm_90")


def jaxlib_include() -> str:
    explicit = os.environ.get("JAXLIB_INCLUDE")
    if explicit:
        if not Path(explicit).is_dir():
            raise SystemExit(f"JAXLIB_INCLUDE={explicit} is not a directory")
        return explicit
    try:
        import jax  # noqa: F401

        return jax.ffi.include_dir()
    except Exception:  # pragma: no cover - build machine without jax
        pass
    try:
        import jaxlib

        return str(Path(jaxlib.__file__).parent / "include")
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "cannot locate the JAX FFI headers; activate an environment with jax "
            f"installed ({exc})"
        )


def newest_source_time() -> float:
    return max(p.stat().st_mtime for p in HEADERS if p.exists())


def needs_build(artifact: Path, force: bool) -> bool:
    if force or not artifact.exists():
        return True
    return artifact.stat().st_mtime < newest_source_time()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def build_cpu(include: str, force: bool, out: Path = OUT) -> Path:
    artifact = out / "libjzradius_cpu.so"
    if not needs_build(artifact, force):
        print(f"up to date: {artifact}")
        return artifact
    out.mkdir(parents=True, exist_ok=True)
    compiler = os.environ.get("CXX", "g++")
    cmd = [
        compiler,
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-fvisibility=default",
        "-I",
        include,
        "-I",
        str(SRC / "cuda"),
        "-o",
        str(artifact),
        *[str(p) for p in CPU_SOURCES],
    ]
    run(cmd)
    return artifact


def nvcc_path() -> str | None:
    explicit = os.environ.get("NVCC")
    if explicit and Path(explicit).exists():
        return explicit
    found = shutil.which("nvcc")
    if found:
        return found
    for candidate in (
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.3/bin/nvcc",
        "/usr/local/cuda-13/bin/nvcc",
    ):
        if Path(candidate).exists():
            return candidate
    return None


def cuda_arches() -> tuple[str, ...]:
    env = os.environ.get("JZRADIUS_CUDA_ARCH")
    if env:
        return tuple(a.strip() for a in env.split(",") if a.strip())
    return DEFAULT_ARCHES


def build_cuda(include: str, force: bool, out: Path = OUT) -> Path | None:
    artifact = out / "libjzradius_cuda.so"
    if not needs_build(artifact, force):
        print(f"up to date: {artifact}")
        return artifact
    nvcc = nvcc_path()
    if nvcc is None:
        print("nvcc not found: skipping the CUDA artifact", file=sys.stderr)
        return None
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        nvcc,
        "-O3",
        "-std=c++17",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        "-Xcompiler",
        "-fvisibility=default",
        "--expt-relaxed-constexpr",
        "-I",
        include,
        "-o",
        str(artifact),
        *[str(p) for p in CUDA_SOURCES],
    ]
    for arch in cuda_arches():
        cmd += ["-gencode", f"arch=compute_{arch[3:]},code={arch}"]
    run(cmd)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="directory for the built artifacts (default: jzradius/_libs)",
    )
    args = parser.parse_args()

    include = jaxlib_include()
    print(f"jaxlib include: {include}")
    cpu = build_cpu(include, args.force, args.out)
    print(f"cpu artifact: {cpu}")
    if not args.no_cuda:
        cuda = build_cuda(include, args.force, args.out)
        print(f"cuda artifact: {cuda}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
