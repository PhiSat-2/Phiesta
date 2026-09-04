from pathlib import Path
import pandas as pd

from phiesta.datasets import PhiestaDataset, build_dataset, normalize_dataset_selection, open_dataset


class FakeEvent:
    def __init__(self, product_id):
        self.product_id = str(product_id)
        self.product_folder = f"/fake/products/{product_id}"
        self.meta = {"path": f"/fake/products/{product_id}/multiband.tif"}

    def georeference(self, output_path=None, **kwargs):
        out = FakeEvent(self.product_id)
        out.product_folder = str(Path(output_path).parent)
        out.meta = {"path": str(output_path), "raster_path": str(output_path), "georeferenced": True}
        return out

    def export_patches(self, out_dir, patch_size=512, prefix="patch", limit=None, **kwargs):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 3 if limit is None else int(limit)
        rows = []
        for i in range(n):
            path = out_dir / f"{prefix}_r0000_c{i:04d}.npy"
            path.write_bytes(b"fake")
            rows.append({"patch_id": f"r0000_c{i:04d}", "patch_path": str(path), "status": "written"})
        return pd.DataFrame(rows)


class FakeClient:
    def __init__(self, fail_ids=None):
        self.fail_ids = set(str(x) for x in (fail_ids or []))
        self.loaded = []

    def _load(self, level, product_id):
        product_id = str(product_id)
        self.loaded.append((level, product_id))
        if product_id in self.fail_ids:
            raise RuntimeError("synthetic load failure")
        return FakeEvent(product_id)

    def load_l1(self, product_id, **kwargs):
        return self._load("L1", product_id)

    def load_l0(self, product_id, **kwargs):
        return self._load("L0", product_id)


def test_normalize_ids_and_deduplicate():
    table = normalize_dataset_selection(["5359", 5360, "5359"])
    assert table["product_id"].tolist() == ["5359", "5360"]


def test_selection_metadata_is_preserved():
    src = pd.DataFrame({"product_id": ["5359"], "label": ["mangrove"], "score": [0.25]})
    out = normalize_dataset_selection(src)
    assert out.iloc[0]["label"] == "mangrove"
    assert out.iloc[0]["score"] == 0.25


def test_build_acquisition_dataset(tmp_path):
    dataset = build_dataset(
        FakeClient(),
        pd.DataFrame({"product_id": ["5359", "5360"], "label": ["a", "b"]}),
        out_dir=tmp_path / "dataset",
        verbose=False,
    )
    assert isinstance(dataset, PhiestaDataset)
    assert len(dataset.acquisitions) == 2
    assert dataset.patches.empty
    assert set(dataset.acquisitions["build_status"]) == {"SUCCESS"}
    assert dataset.acquisitions["label"].tolist() == ["a", "b"]
    assert dataset.acquisition_manifest_path.exists()


def test_patch_dataset_propagates_metadata(tmp_path):
    selection = pd.DataFrame({"product_id": ["5359"], "label": ["mangrove"], "group": ["coast"]})
    dataset = build_dataset(
        FakeClient(), selection, out_dir=tmp_path / "dataset",
        patch_size=256, patch_limit=2, verbose=False,
    )
    assert len(dataset.patches) == 2
    assert dataset.patches["label"].tolist() == ["mangrove", "mangrove"]
    assert dataset.patches["group"].tolist() == ["coast", "coast"]
    assert dataset.patches["dataset_patch_id"].is_unique
    assert int(dataset.acquisitions.iloc[0]["patch_count"]) == 2


def test_continue_after_failure(tmp_path):
    dataset = build_dataset(
        FakeClient(fail_ids={"5360"}), ["5359", "5360", "5361"],
        out_dir=tmp_path / "dataset", verbose=False,
    )
    status = dict(zip(dataset.acquisitions["product_id"].astype(str), dataset.acquisitions["build_status"]))
    assert status == {"5359": "SUCCESS", "5360": "FAILED", "5361": "SUCCESS"}


def test_resume_skips_success(tmp_path):
    root = tmp_path / "dataset"
    first = FakeClient()
    build_dataset(first, ["5359"], out_dir=root, verbose=False)
    second = FakeClient()
    dataset = build_dataset(second, ["5359", "5360"], out_dir=root, resume=True, verbose=False)
    assert second.loaded == [("L1", "5360")]
    assert set(dataset.acquisitions["product_id"].astype(str)) == {"5359", "5360"}


def test_georeferenced_dataset(tmp_path):
    dataset = build_dataset(FakeClient(), ["5359"], out_dir=tmp_path / "dataset", georeference=True, verbose=False)
    row = dataset.acquisitions.iloc[0]
    assert bool(row["georeferenced"]) is True
    assert Path(str(row["raster_path"])).as_posix().endswith("georeferenced/phisat2_5359_georeferenced.tif")


def test_open_dataset(tmp_path):
    root = tmp_path / "dataset"
    built = build_dataset(FakeClient(), ["5359"], out_dir=root, patch_size=128, patch_limit=1, verbose=False)
    reopened = open_dataset(root)
    assert len(reopened.acquisitions) == len(built.acquisitions)
    assert len(reopened.patches) == len(built.patches)
    assert reopened.metadata["format"] == "phiesta-dataset-v1"

