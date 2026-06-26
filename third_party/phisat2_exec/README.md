# Optional local executables

This directory is intentionally kept without executable binaries.

The public Python simulator code used by Phiesta is included in:

    third_party/orbitalai_phisat2_sim/

This folder is only a placeholder for optional local executable binaries or platform-specific helpers that are not distributed with this repository.

If you have authorized access to such binaries, place them locally here or pass their path explicitly through the API.

Examples of local-only files:

    third_party/phisat2_exec/phisat2_unix.bin
    third_party/phisat2_exec/phisat2_win.bin
    third_party/phisat2_exec/phisat2_osx-arm64.bin
    third_party/phisat2_exec/phisat2_osx-x86_64.bin

These files are ignored by Git and should not be committed.
