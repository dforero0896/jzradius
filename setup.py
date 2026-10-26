"""Build hook for jzradius: compile the FFI libraries, then tag the wheel.

jzradius does not build a CPython extension module.  Its kernels are plain
C-ABI shared objects that are registered with ``jax.ffi.register_ffi_target``
and loaded through ``ctypes``; the only thing they need from the host language
is a path and a ``PyCapsule``.  Setuptools' compiler machinery has no idea how
to invoke nvcc or where the XLA FFI headers live, so this file delegates the
native build to ``scripts/build_ext.py`` -- the same driver the test fixtures and
the documented workflow use, so there is exactly one build recipe -- and only
teaches setuptools two things: run that driver before ``build_py`` copies the
artifacts, and give the wheel a platform tag instead of ``py3-none-any``
(the payload is ``linux_x86_64`` shared objects, but it does not link libpython,
so the tag stays ``py3-none-<plat>``).

Environment variables
---------------------
``JZRADIUS_SKIP_NATIVE=1``
    Install without native libraries.  ``import jzradius`` still works; the FFI
    raises ``FfiUnavailable`` until the libraries are built.
``JZRADIUS_BUILD_CUDA=1``
    Also compile the CUDA artifact (requires nvcc and a CUDA toolkit whose
    major version matches the installed JAX).  The default is the CPU mirror
    only, because a wheel built on a login node should not guess GPU arches.
``JAXLIB_INCLUDE=<dir>``
    Path to the XLA FFI headers, for environments where ``jax`` is not
    importable at build time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build import build as _build

ROOT = Path(__file__).resolve().parent
LIBS = ROOT / "jzradius" / "_libs"

try:  # wheel is declared in [build-system].requires, but stay importable without it
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except Exception:  # pragma: no cover - sdist/bdist only
    _bdist_wheel = None


class build(_build):
    """Compile into a private staging dir, then hand the artifacts to build_py.

    The artifacts are *not* declared as package data on purpose: a checkout
    normally contains developer-built libraries in ``jzradius/_libs`` (including
    a CUDA one compiled for whatever toolkit the login node had), and a wheel
    must only ever ship what this build produced.
    """

    def run(self) -> None:  # noqa: D102 - setuptools command
        super().run()
        self._build_native()

    def _build_native(self) -> None:
        target = Path(self.build_lib) / "jzradius" / "_libs"
        target.mkdir(parents=True, exist_ok=True)
        # build_py is incremental and `build/` persists between invocations, so
        # drop whatever an earlier build left here *before* deciding what to
        # bundle: a stale artifact must not survive a CPU-only or a
        # JZRADIUS_SKIP_NATIVE=1 build.
        for stale in target.glob("*.so"):
            stale.unlink()

        if os.environ.get("JZRADIUS_SKIP_NATIVE") == "1":
            print(
                "jzradius: JZRADIUS_SKIP_NATIVE=1 -- installing without native "
                "libraries; the FFI will raise FfiUnavailable until they exist",
            )
            return
        staging = Path(self.build_base) / "native"
        build_cuda = os.environ.get("JZRADIUS_BUILD_CUDA") == "1"
        if not build_cuda:
            # Do not resurrect a CUDA artifact from an earlier build here.
            (staging / "libjzradius_cuda.so").unlink(missing_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "build_ext.py"),
            "--out",
            str(staging),
        ]
        if not build_cuda:
            cmd.append("--no-cuda")
        print("jzradius: " + " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, cwd=ROOT)
        except subprocess.CalledProcessError:
            # A missing compiler or missing XLA headers must not make the
            # package uninstallable: the Python layer is useful on its own
            # (reference implementation, bounds, CPU traversal) and ffi.py
            # explains what to run once a toolchain is available.
            print(
                "jzradius: WARNING: the native build failed; installing without "
                "the FFI libraries. Rebuild later with `python "
                "scripts/build_ext.py --force` or set JAXLIB_INCLUDE.",
                file=sys.stderr,
            )
            return

        artifacts = sorted(staging.glob("*.so"))
        for artifact in artifacts:
            shutil.copy2(artifact, target / artifact.name)
        # Editable installs (and the documented `scripts/build_ext.py` workflow)
        # read the libraries straight out of the source tree, where build_py
        # never writes; mirror the fresh artifacts there.
        LIBS.mkdir(parents=True, exist_ok=True)
        for artifact in artifacts:
            shutil.copy2(artifact, LIBS / artifact.name)
        if artifacts:
            print(
                "jzradius: bundled " + ", ".join(a.name for a in artifacts),
            )
        else:
            print(
                "jzradius: WARNING: no native artifacts were produced; the "
                "wheel will raise FfiUnavailable until they are built",
                file=sys.stderr,
            )


if _bdist_wheel is not None:

    class bdist_wheel(_bdist_wheel):
        """Non-pure wheel that is still ABI independent."""

        def finalize_options(self) -> None:
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self) -> tuple[str, str, str]:
            _, _, plat = super().get_tag()
            return "py3", "none", plat


class BinaryDistribution(Distribution):
    """Force the platform-specific install layout and wheel tag."""

    def has_ext_modules(self) -> bool:
        return True


cmdclass = {"build": build}
if _bdist_wheel is not None:
    cmdclass["bdist_wheel"] = bdist_wheel

setup(cmdclass=cmdclass, distclass=BinaryDistribution)
