#!/usr/bin/env python3
"""M0 environment probe (PLAN.md 4.3).

Prints the environment facts required by PLAN.md 3.3/4.3 and optionally writes
them to a markdown file. The ``jax.lib.xla_extension`` query in the plan's
snippet no longer exists in JAX 0.10.2, so it is guarded (see Deviation D001).

On a login node the CUDA JAX plugin can stall, so CPU platform is forced
unless ``--gpu`` is passed.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _clean_sys_path() -> None:
    """Drop CWD so the local ``jztree/`` clone cannot shadow an installed pkg."""
    cwd = os.getcwd()
    sys.path[:] = [p for p in sys.path if p not in ("", cwd)]


def _run(cmd: list[str], timeout: float = 10.0):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:  # noqa: BLE001
        return None, "", str(exc)


def _find_nvcc() -> str | None:
    found = shutil.which("nvcc")
    if found:
        return found
    for candidate in (
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.3/bin/nvcc",
        "/usr/local/cuda-13/bin/nvcc",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="require a GPU backend")
    parser.add_argument("--write", type=Path, default=None,
                        help="write the markdown report here")
    parser.add_argument("--nvidia-smi-out", type=Path,
                        default=Path("docs/nvidia_smi.txt"))
    args = parser.parse_args()

    if not args.gpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    _clean_sys_path()

    import numpy as np  # noqa: PLC0415

    import jax  # noqa: PLC0415
    import jaxlib  # noqa: PLC0415

    try:  # JAX <= 0.4 style; removed in 0.10.x (Deviation D001).
        xla_version = jax.lib.xla_extension.version()
    except AttributeError:
        xla_version = None

    devices = [str(d) for d in jax.devices()]
    backend = jax.default_backend()

    jztree_error = None
    jztree_version = None
    jztree_file = None
    try:
        import jztree  # noqa: PLC0415

        jztree_version = getattr(jztree, "__version__", "unknown")
        jztree_file = getattr(jztree, "__file__", None)
        # A namespace-package hit means the clone, not an install, was imported.
        if jztree_file is None:
            jztree_error = "namespace package only (not installed)"
    except Exception as exc:  # noqa: BLE001
        jztree_error = f"{type(exc).__name__}: {exc}"

    nvcc = _find_nvcc()
    nvcc_release = None
    if nvcc:
        _, out, _ = _run([nvcc, "--version"])
        for token in out.split():
            if token.startswith("V") and token[1:2].isdigit():
                nvcc_release = token
                break

    smi = shutil.which("nvidia-smi")
    smi_first_line = None
    if smi:
        rc, out, _ = _run([smi, "-L"])
        if rc == 0 and out:
            smi_first_line = out.splitlines()[0]
            args.nvidia_smi_out.parent.mkdir(parents=True, exist_ok=True)
            args.nvidia_smi_out.write_text(out + "\n")

    slurm_gpu = None
    sinfo = shutil.which("sinfo")
    if sinfo:
        rc, out, _ = _run(["sinfo", "-h", "-p", "gpu", "-o", "%n %G %t"])
        if rc == 0:
            slurm_gpu = out

    lines = [
        "# Environment lock (generated)",
        "",
        f"- host: `{platform.node()}` ({platform.system()} {platform.release()})",
        f"- python: `{sys.version.split()[0]}` (`{sys.executable}`)",
        f"- jax: `{jax.__version__}`",
        f"- jaxlib: `{jaxlib.__version__}`",
        f"- jax.lib.xla_extension version: `{xla_version}`",
        f"- numpy: `{np.__version__}`",
        f"- jax backend: `{backend}`",
        f"- devices: {devices}",
        f"- jztree: version=`{jztree_version}` file=`{jztree_file}` "
        f"error={jztree_error}",
        f"- nvcc: `{nvcc}` release `{nvcc_release}`",
        f"- nvidia-smi: `{smi_first_line}`",
        f"- SLURM gpu partition: `{slurm_gpu}`",
        f"- CUDA_VISIBLE_DEVICES: `{os.environ.get('CUDA_VISIBLE_DEVICES')}`",
        "",
    ]
    report = "\n".join(lines)
    print(report)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(report)

    if args.gpu and backend != "gpu":
        print("ERROR: --gpu requested but JAX backend is not gpu", file=sys.stderr)
        return 2
    if args.gpu and not smi_first_line:
        print("ERROR: --gpu requested but nvidia-smi sees no GPU", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
