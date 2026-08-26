# Contributing to Phiesta

Contributions are welcome through GitHub issues and pull requests.

## Development setup

```bash
git clone https://github.com/PhiSat-2/Phiesta.git
cd Phiesta
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install pytest
```

For the optional Sentinel-2 / strict-georeferencing stack:

```bash
python -m pip install -e ".[triplets]"
python -m pip install git+https://github.com/cvg/LightGlue.git
```

## Before opening a pull request

```bash
python -m compileall -q phiesta
pytest
```

Do not commit credentials, downloaded mission data, generated outputs, or machine-specific notebook outputs. New third-party code or binaries must include documented provenance and redistribution terms.
