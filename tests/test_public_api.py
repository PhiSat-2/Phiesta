import phiesta


def test_documented_event_classes_are_public():
    assert phiesta.L0_event is not None
    assert phiesta.L1_event is not None
    assert phiesta.L1A_event is not None


def test_mission_band_table():
    table = phiesta.phisat2_band_table()
    assert len(table) == 8
    assert set(table["role"]) >= {"blue", "green", "red", "nir", "panchromatic"}


def test_product_level_specs_include_l1a_l1c():
    specs = phiesta.phisat2_product_level_specs()
    assert {"L1A", "L1C"}.issubset(set(specs["level"]))
