from getpass import os
import getpass

from .insula_client import InsulaClient
from .constants import PHISAT2_BASE_URL

from ..sys_cfg import DATA_PATH

def _load_insula_openid_connect():
    try:
        from InsulaWorkflowClient import InsulaOpenIDConnect
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "InsulaWorkflowClient is required to use connect_insula(). "
            "Install it in the current Python environment, or make sure it is bundled "
            "with Phiesta."
        ) from exc

    return InsulaOpenIDConnect

def connect_insula(
    username: str | None = None,
    password: str | None = None,
    base_url: str = PHISAT2_BASE_URL,
    cache_dir: str | None = None,
) -> InsulaClient:
    """
    Create and return an authenticated `InsulaClient` for the PHISAT-2 Insula instance.

    This helper is the recommended entry point for most users. It prompts for
    credentials when they are not provided explicitly.

    Args:
        username: Insula username or email. If `None`, prompt interactively.
        password: Insula password. If `None`, prompt interactively.
        base_url: Base URL of the Insula deployment.
        cache_dir: Optional local cache/download root. If `None`, use the package
            default data directory.

    Returns:
        An authenticated `InsulaClient`.
    """

    InsulaOpenIDConnect = _load_insula_openid_connect()
    insula_auth = InsulaOpenIDConnect(
        authorization_endpoint="https://identity.insula.earth/realms/phisat2/protocol/openid-connect/auth",
        token_endpoint="https://identity.insula.earth/realms/phisat2/protocol/openid-connect/token",
        redirect_uri="http://localhost:9207/auth",
        client_id="api-client",
    )

    if username is None:
        username = username or os.environ.get("INSULA_USERNAME") or os.environ.get("INSULA_EMAIL")
    if username is None:
        username = input("Insula username/email: ")
    if password is None:
        password = getpass.getpass("Insula password: ")

    insula_auth.set_user_credentials(username=username, password=password)

    return InsulaClient(
        insula_auth=insula_auth,
        base_url=base_url,
        cache_dir=cache_dir or DATA_PATH,
    )