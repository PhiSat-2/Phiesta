import json
from pathlib import Path


def test_quickstart_notebook_is_current_and_safe():
    path = Path("examples/Phiesta_Quickstart.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))

    sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    ]
    text = "\n".join(sources)

    assert "RUN_FULL_TRIPLET = False" in text
    assert "RUN_FULL_TRIPLET = True" not in text
    assert "RUN_GEOREFERENCE = False" in text
    assert "RUN_DATASET_WORKFLOW = False" in text
    assert "RUN_WORLDCOVER_SEARCH = False" in text

    for api_name in (
        "search_l1_worldcover",
        "georeference(",
        "build_l1_dataset",
        "make_splits",
        "add_target",
        "to_dataloader",
    ):
        assert api_name in text

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
