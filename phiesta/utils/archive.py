from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:$")


def _validated_member_target(
    destination: Path,
    info: zipfile.ZipInfo,
) -> Path:
    """Validate one ZIP member before extraction."""
    raw_name = str(info.filename)

    if "\x00" in raw_name:
        raise ValueError(
            f"Unsafe ZIP member contains NUL byte: {raw_name!r}"
        )

    normalized = raw_name.replace("\\", "/")

    if normalized.startswith("/"):
        raise ValueError(
            f"Unsafe absolute ZIP member path: {raw_name!r}"
        )

    pure = PurePosixPath(normalized)
    parts = pure.parts

    if parts and _DRIVE_PREFIX.match(parts[0]):
        raise ValueError(
            f"Unsafe drive-qualified ZIP member path: {raw_name!r}"
        )

    if any(part == ".." for part in parts):
        raise ValueError(
            f"Unsafe parent traversal in ZIP member: {raw_name!r}"
        )

    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(unix_mode):
        raise ValueError(
            f"Unsafe symbolic-link ZIP member: {raw_name!r}"
        )

    destination = destination.resolve()
    target = destination.joinpath(*parts).resolve()

    try:
        target.relative_to(destination)
    except ValueError as exc:
        raise ValueError(
            f"Unsafe ZIP member escapes destination: {raw_name!r}"
        ) from exc

    return target


def safe_extract_zip(
    archive: str | Path | zipfile.ZipFile,
    destination: str | Path,
) -> Path:
    """
    Extract a ZIP archive only after validating every member path.

    Validation happens for the complete archive before extraction starts, so a
    rejected archive cannot leave a partially extracted tree behind.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()

    owns_archive = not isinstance(archive, zipfile.ZipFile)
    zf = (
        zipfile.ZipFile(Path(archive), "r")
        if owns_archive
        else archive
    )

    try:
        members = zf.infolist()

        for info in members:
            _validated_member_target(destination, info)

        zf.extractall(destination)
    finally:
        if owns_archive:
            zf.close()

    return destination
