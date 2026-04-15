from __future__ import annotations

from pathlib import Path
from getpass import getpass

from InsulaWorkflowClient import InsulaOpenIDConnect

from pyrawph.remote import InsulaClient


PHISAT2_L1_COLLECTION = "phisat24e55ba83dd304ea9b018b65e9b17a7de"
BASE_URL = "https://phisat2.insula.earth"


def build_auth():
    insula_auth = InsulaOpenIDConnect(
        authorization_endpoint="https://identity.insula.earth/realms/phisat2/protocol/openid-connect/auth",
        token_endpoint="https://identity.insula.earth/realms/phisat2/protocol/openid-connect/token",
        redirect_uri="http://localhost:9207/auth",
        client_id="api-client",
    )

    username = input("Insula username/email: ")
    password = getpass("Insula password: ")
    insula_auth.set_user_credentials(username=username, password=password)
    return insula_auth


def main():
    dest_dir = Path(input("Destination cache dir: ").strip() or "./insula_cache")
    max_pages_raw = input("Max pages (blank = all): ").strip()
    max_pages = int(max_pages_raw) if max_pages_raw else None

    insula_auth = build_auth()
    client = InsulaClient(
        insula_auth=insula_auth,
        base_url=BASE_URL,
        cache_dir=dest_dir,
    )

    n = 0
    for feature in client.iter_ref_data(
        ref_data_collection=PHISAT2_L1_COLLECTION,
        results_per_page=50,
        max_pages=max_pages,
    ):
        props = feature["properties"]
        identifier = props.get("productIdentifier", "unknown")
        print(f"[{n}] Downloading {identifier}")

        try:
            product_folder = client.download_feature(
                feature,
                extract=True,
                keep_zip=False,
                skip_existing=True,
            )
            print(f"    -> {product_folder}")
            n += 1
        except Exception as exc:
            print(f"    !! failed: {exc}")

    print(f"\nDone. Downloaded or reused {n} product(s).")


if __name__ == "__main__":
    main()