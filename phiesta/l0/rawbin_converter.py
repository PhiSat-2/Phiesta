from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional
import tempfile


def _default_sim_root() -> Path:
    """
    Resolve the bundled SimToTiff root directory.

    Resolution order:
      1) PYRAWPH_SIM_ROOT environment variable
      2) bundled location under third_party/simtotiff/

    Returns:
        Path to the converter root containing simera/packet_to_tiff.py.

    Raises:
        FileNotFoundError: If no valid converter root can be found.
    """
    env_root = os.environ.get("PYRAWPH_SIM_ROOT")
    if env_root:
        p = Path(env_root)
        if (p / "simera" / "packet_to_tiff.py").exists():
            return p
        raise FileNotFoundError(
            f"PYRAWPH_SIM_ROOT is set but does not contain simera/packet_to_tiff.py: {p}"
        )

    repo_root = Path(__file__).resolve().parents[2]
    bundled = (
        repo_root
        / "third_party"
        / "simtotiff"
        / "SA072-SENSE-Conversion-Code-main"
        / "SA072-SENSE-Conversion-Code-main"
    )

    if (bundled / "simera" / "packet_to_tiff.py").exists():
        return bundled

    raise FileNotFoundError(
        "Could not resolve a bundled SimToTiff root. "
        "Set PYRAWPH_SIM_ROOT or place the converter under "
        "third_party/simtotiff/SA072-SENSE-Conversion-Code-main/"
        "SA072-SENSE-Conversion-Code-main/."
    )


def _safe_rmtree(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


def _safe_unlink(f: Path) -> None:
    if f.exists():
        f.unlink()


def convert_l0_rawbin_inplace(
    product_folder: str | Path,
    sim_root: str | Path | None = None,
    simera_subdir: str = "simera",
    packet_script: str = "packet_to_tiff.py",
    python_exe: Optional[str] = None,
    overwrite_raw: bool = True,
) -> Path:
    """
    Convert an Insula-downloaded L0 product containing raw.bin into a PyRawPh-readable
    folder by generating raw TIFF bands in product_folder/raw.

    The converter is executed in an isolated temporary workspace to avoid collisions
    between concurrent conversions.

    Args:
        product_folder: Path to the L0 product folder containing raw.bin.
        sim_root: Optional converter root. If None, use the bundled/default resolver.
        simera_subdir: Subdirectory inside the converter root containing packet_to_tiff.py.
        packet_script: Name of the conversion script to execute.
        python_exe: Optional Python executable to use. Defaults to sys.executable.
        overwrite_raw: If True, overwrite an existing product_folder/raw directory.

    Returns:
        The same product folder, now augmented with a raw/ directory.

    Raises:
        FileNotFoundError: If required inputs or converter files are missing.
        RuntimeError: If conversion fails or produces no raw/ output.
    """
    product_folder = Path(product_folder)
    sim_root = Path(sim_root) if sim_root is not None else _default_sim_root()
    python_exe = python_exe or sys.executable

    raw_bin_src = product_folder / "raw.bin"
    if not raw_bin_src.exists():
        raise FileNotFoundError(f"Missing raw.bin in {product_folder}")

    if not sim_root.is_dir():
        raise FileNotFoundError(
            f"sim_root not found: {sim_root}\n"
            f"Either bundle the converter in third_party/simtotiff/..., "
            f"or pass sim_root=..., or define PYRAWPH_SIM_ROOT."
        )

    if not (sim_root / simera_subdir).is_dir():
        raise FileNotFoundError(f"SIMERA_DIR not found: {sim_root / simera_subdir}")

    if not (sim_root / simera_subdir / packet_script).exists():
        raise FileNotFoundError(f"{packet_script} not found in {sim_root / simera_subdir}")

    dest_raw_dir = product_folder / "raw"
    if dest_raw_dir.exists():
        if overwrite_raw:
            shutil.rmtree(dest_raw_dir)
        else:
            return product_folder

    with tempfile.TemporaryDirectory(prefix="pyrawph_l0conv_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        work_root = tmp_root / "converter"

        # Copy the converter tree into an isolated temporary workspace
        shutil.copytree(sim_root, work_root)

        work_simera_dir = work_root / simera_subdir
        work_raw_bin = work_root / "raw.bin"
        work_raw_dir = work_root / "raw"

        shutil.copy2(raw_bin_src, work_raw_bin)

        completed = subprocess.run(
            [python_exe, packet_script],
            cwd=str(work_simera_dir),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{packet_script} failed with return code {completed.returncode}")

        if not work_raw_dir.exists() or not any(work_raw_dir.iterdir()):
            raise RuntimeError(f"Conversion finished but {work_raw_dir} is missing or empty.")

        shutil.copytree(work_raw_dir, dest_raw_dir)

    return product_folder