import pytest

from phiesta.remote.insula_client import (
    _platform_file_id_from_href,
    _resolve_feature_download_url,
)


def test_platform_file_id_from_catalogue_href():
    assert (
        _platform_file_id_from_href(
            "https://phisat2.insula.earth/secure/api/v2.0/"
            "search/products/platform/20731"
        )
        == "20731"
    )


def test_resolve_feature_download_url_prefers_direct_download():
    feature = {
        "properties": {
            "_links": {
                "download": {
                    "href": "https://example.test/direct.zip"
                },
                "platformFile": {
                    "href": (
                        "https://phisat2.insula.earth/secure/api/v2.0/"
                        "search/products/platform/20731"
                    )
                },
            }
        }
    }

    assert _resolve_feature_download_url(
        feature,
        base_url="https://phisat2.insula.earth",
    ) == "https://example.test/direct.zip"


def test_resolve_feature_download_url_from_platform_file():
    feature = {
        "properties": {
            "_links": {
                "platformFile": {
                    "href": (
                        "https://phisat2.insula.earth/secure/api/v2.0/"
                        "search/products/platform/20731"
                    )
                },
                "platform": {
                    "href": "eopaas://refData/8/example.zip"
                },
            },
            "services": {
                "download": {
                    "url": (
                        "http://eopaas-resto-phisat2/resto/"
                        "collections/example/download"
                    )
                }
            },
        }
    }

    assert _resolve_feature_download_url(
        feature,
        base_url="https://phisat2.insula.earth",
    ) == (
        "https://phisat2.insula.earth/secure/api/v2.0/"
        "platformFiles/20731/dl"
    )


def test_resolve_feature_download_url_rejects_internal_only_service():
    feature = {
        "properties": {
            "_links": {
                "platform": {
                    "href": "eopaas://refData/8/example.zip"
                }
            },
            "services": {
                "download": {
                    "url": "http://eopaas-resto-phisat2/internal/download"
                }
            },
        }
    }

    with pytest.raises(ValueError, match="No externally usable"):
        _resolve_feature_download_url(
            feature,
            base_url="https://phisat2.insula.earth",
        )
