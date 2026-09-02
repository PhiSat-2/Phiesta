from __future__ import annotations

from datetime import datetime

import numpy as np

from phiesta.triplets.sentinel_crop import (
    _dn_to_toa_reflectance,
    _earth_sun_correction_u,
    _extract_simulator_metadata,
)


def test_dn_to_toa_reflectance_preserves_nodata_and_applies_offset():
    dn = np.array([[0, 1000, 5000, 10000]], dtype=np.float32)
    got = _dn_to_toa_reflectance(
        dn,
        band="B02",
        quantification_value=10000.0,
        radiometric_offsets={"B02": -1000.0},
    )
    expected = np.array([[0.0, 0.0, 0.4, 0.9]], dtype=np.float32)
    assert np.allclose(got, expected)


def test_extract_simulator_metadata_reads_namespaced_l1c_radiometry(tmp_path):
    safe = tmp_path / "S2B_TEST.SAFE"
    granule = safe / "GRANULE" / "L1C_T31_TEST"
    granule.mkdir(parents=True)

    irradiances = {
        1: 1959.0,
        2: 1824.0,
        3: 1512.0,
        4: 1425.0,
        5: 1288.0,
        6: 1163.0,
        7: 1036.0,
    }
    irr_xml = "\n".join(
        f'<n:SOLAR_IRRADIANCE bandId="{band_id}">{value}</n:SOLAR_IRRADIANCE>'
        for band_id, value in irradiances.items()
    )
    offset_xml = "\n".join(
        f'<n:RADIO_ADD_OFFSET band_id="{band_id}">-1000</n:RADIO_ADD_OFFSET>'
        for band_id in irradiances
    )

    (safe / "MTD_MSIL1C.xml").write_text(
        f"""<?xml version="1.0"?>
<n:Product xmlns:n="urn:test">
  <n:QUANTIFICATION_VALUE>10000</n:QUANTIFICATION_VALUE>
  <n:Reflectance_Conversion>
    <n:U>1.0342</n:U>
    <n:Solar_Irradiance_List>{irr_xml}</n:Solar_Irradiance_List>
  </n:Reflectance_Conversion>
  <n:Radiometric_Offset_List>{offset_xml}</n:Radiometric_Offset_List>
</n:Product>
""",
        encoding="utf-8",
    )
    (granule / "MTD_TL.xml").write_text(
        """<?xml version="1.0"?>
<n:Tile xmlns:n="urn:test">
  <n:Sun_Angles_Grid>
    <n:Zenith>
      <n:Values_List>
        <n:VALUES>20 21</n:VALUES>
        <n:VALUES>22 23</n:VALUES>
      </n:Values_List>
    </n:Zenith>
  </n:Sun_Angles_Grid>
</n:Tile>
""",
        encoding="utf-8",
    )

    meta = _extract_simulator_metadata(safe, "2026-01-03T10:00:00Z")

    assert meta["crop_value_domain"] == "toa_reflectance"
    assert meta["radiometry_version"] == 1
    assert meta["quantification_value"] == 10000.0
    assert meta["reflectance_conversion_U"] == 1.0342
    assert meta["earth_sun_dist"] == 1.0342
    assert meta["radiometric_offsets"]["B02"] == -1000.0
    assert meta["radiometric_offsets"]["B08"] == -1000.0
    assert meta["solar_irradiances"]["B02"] == 1959.0
    assert meta["solar_irradiances"]["B08"] == 1036.0
    assert meta["sun_zenith_angles"] == [[20.0, 21.0], [22.0, 23.0]]


def test_u_fallback_is_inverse_squared_earth_sun_distance():
    u = _earth_sun_correction_u(datetime(2026, 1, 4))
    assert 1.02 < u < 1.05
