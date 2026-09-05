from pathlib import Path
import stat
import zipfile

import pytest

from phiesta.utils.archive import safe_extract_zip


def _write_zip(path: Path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members:
            zf.writestr(name, content)


def test_safe_extract_zip_extracts_normal_archive(tmp_path):
    archive = tmp_path / "ok.zip"
    out = tmp_path / "out"

    _write_zip(
        archive,
        [
            ("product/metadata.json", "{}"),
            ("product/bands/b1.tif", "fake"),
        ],
    )

    result = safe_extract_zip(archive, out)

    assert result == out.resolve()
    assert (out / "product" / "metadata.json").read_text() == "{}"
    assert (out / "product" / "bands" / "b1.tif").read_text() == "fake"


@pytest.mark.parametrize(
    "member",
    [
        "../outside.txt",
        "product/../../outside.txt",
        r"..\outside.txt",
        "/absolute.txt",
        "C:/absolute.txt",
        r"C:\absolute.txt",
    ],
)
def test_safe_extract_zip_rejects_path_escape(tmp_path, member):
    archive = tmp_path / "bad.zip"
    out = tmp_path / "out"

    _write_zip(archive, [(member, "bad")])

    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract_zip(archive, out)

    assert not (tmp_path / "outside.txt").exists()


def test_safe_extract_zip_rejects_symlink_member(tmp_path):
    archive = tmp_path / "symlink.zip"
    out = tmp_path / "out"

    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16

    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(info, "../outside")

    with pytest.raises(ValueError, match="symbolic-link"):
        safe_extract_zip(archive, out)
