from phiesta.remote.insula_client import _resolve_feature_download_url


def test_platform_file_download_url_is_resolved_for_catalogued_product():
    feature = {
        "properties": {
            "filename": "PHISAT-2_L1A_test.zip",
            "platformUsable": False,
            "_links": {
                "platformFile": {
                    "href": (
                        "https://phisat2.insula.earth/secure/api/v2.0/"
                        "search/products/platform/20731"
                    )
                }
            },
        }
    }

    assert _resolve_feature_download_url(
        feature,
        base_url="https://phisat2.insula.earth",
    ).endswith("/platformFiles/20731/dl")
