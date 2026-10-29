# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Install a wheel in a fresh venv and exercise the CLI/WebUI outside the repo."""

import argparse
import json
import os
import socket
import subprocess
import tempfile
import time
import venv
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def check_distribution(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "multi_modal_ai_studio/config/presets/speaches-jetson.yaml" in names
        assert "multi_modal_ai_studio/webui/static/index.html" in names
        assert not any(name.endswith(".backup") for name in names)
    env = {
        key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")
    }
    with tempfile.TemporaryDirectory(prefix="mmas-wheel-smoke-") as temporary:
        root = Path(temporary)
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        binary_dir = root / "venv" / ("Scripts" if os.name == "nt" else "bin")
        python = binary_dir / ("python.exe" if os.name == "nt" else "python")
        cli = binary_dir / (
            "multi-modal-ai-studio.exe" if os.name == "nt" else "multi-modal-ai-studio"
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", str(wheel)],
            cwd=root,
            env=env,
            check=True,
            timeout=600,
        )
        subprocess.run([str(cli), "--version"], cwd=root, env=env, check=True)
        subprocess.run(
            [str(cli), "--help"], cwd=root, env=env, check=True, stdout=subprocess.DEVNULL
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                (
                    "import multi_modal_ai_studio as m; print(m.__file__); "
                    "from pathlib import Path; import sys; "
                    "assert Path(m.__file__).is_relative_to(Path(sys.prefix))"
                ),
            ],
            cwd=root,
            env=env,
            check=True,
        )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        with (root / "server.log").open("w+") as log:
            process = subprocess.Popen(
                [
                    str(cli),
                    "--preset",
                    "speaches-jetson",
                    "--no-ssl",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--session-dir",
                    str(root / "sessions"),
                ],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Installed CLI exited before serving HTTP")
                    try:
                        with urlopen(base + "/api/config/initial", timeout=2) as response:
                            initial = json.load(response)
                        break
                    except (URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.1)
                assert initial["llm"]["model"] == "gemma-4-e2b"
                with urlopen(base + "/api/presets", timeout=5) as response:
                    listed = json.load(response)["presets"]
                assert "speaches-jetson.yaml" in {item["filename"] for item in listed}
                with urlopen(base + "/api/presets/speaches-jetson", timeout=5) as response:
                    assert json.load(response) == initial
                for route in ("/", "/static/app.js", "/static/styles.css"):
                    with urlopen(base + route, timeout=5) as response:
                        assert response.status == 200
                        assert response.read()
                print("PASS: clean wheel installation, CLI preset, preset API, and static UI")
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read())
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    check_distribution(args.wheel.resolve(strict=True))
