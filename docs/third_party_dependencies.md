# Third-Party Dependencies and Licenses (OSRB / nSpect)

This doc describes how to produce a list of 3rd party dependencies with license information for NVIDIA Open Source Review Board (OSRB) and security inspection (e.g. nSpect).

## Generating the list in this repo

### Option 1: Script (recommended)

From the repo root (no venv or activation needed):

```bash
chmod +x scripts/generate_third_party_licenses.sh
./scripts/generate_third_party_licenses.sh

# Generate separate optional-runtime manifests when those features ship.
./scripts/generate_third_party_licenses.sh video-input
./scripts/generate_third_party_licenses.sh audio,camera,video-input,webrtc-camera
```

The script uses a temporary virtual environment so it works even when system Python is externally managed (PEP 668). It uses [pip-licenses](https://pypi.org/project/pip-licenses/) to write:

- `third_party_licenses.csv` – core environment for spreadsheets / nSpect submission
- `third_party_licenses.md` – core human-readable table
- `third_party_licenses-<profile>.*` – separately generated optional environments

### Option 2: Manual pip-licenses

```bash
pip install -e .
pip install pip-licenses
pip-licenses --format=csv -o third_party_licenses.csv
pip-licenses --format=markdown -o third_party_licenses.md
```

### Declared dependencies (pyproject.toml)

`pyproject.toml` is the single source of truth for runtime and optional
dependencies. Core, audio, camera, Video Input, and WebRTC-camera environments
must be reviewed separately because optional image/video parsers are not part
of a speech-only installation. The generated list includes all transitive
dependencies of the installed environment, not only the direct declarations.

## License of this project

This project is licensed under **Apache-2.0**.
