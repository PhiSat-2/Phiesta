from phiesta.l1.l1_event import L1_event


def test_georeference_forwards_high_level_options(monkeypatch):
    event = object.__new__(L1_event)
    captured = {}

    def fake_export(self, output_path=None, **kwargs):
        captured["output_path"] = output_path
        captured["kwargs"] = kwargs
        return {"status": "SUCCESS", "path": "out.tif"}

    monkeypatch.setattr(L1_event, "export_georeferenced_tif", fake_export)

    result = event.georeference(
        output_path="custom.tif",
        window_days=10,
        sentinel_backend="download",
        verbose=False,
    )

    assert result["path"] == "out.tif"
    assert captured == {
        "output_path": "custom.tif",
        "kwargs": {
            "window_days": 10,
            "sentinel_backend": "download",
            "verbose": False,
        },
    }
