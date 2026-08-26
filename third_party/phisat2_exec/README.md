# ΦSat-2 simulator executables

This directory contains optional platform-specific ΦSat-2 simulator executables used by Phiesta's simulation backend.

They are repository assets rather than Python source code and are intentionally **not included in the Python wheel** by ``pyproject.toml``.

The Apache-2.0 license at the root of Phiesta does **not** automatically apply to these binaries. Their use and redistribution are governed by the terms or authorization of their rights holder. See ``../../THIRD_PARTY_NOTICES.md``.

Expected filenames include:

- ``phisat2_unix.bin``
- ``phisat2_win.bin``
- ``phisat2_osx-arm64.bin``
- ``phisat2_osx-x86_64.bin``
