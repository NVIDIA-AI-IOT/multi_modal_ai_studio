# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Install a wheel in a fresh venv and exercise the CLI/WebUI outside the repo."""

import argparse
import configparser
import json
import os
import socket
import subprocess
import tempfile
import time
import venv
import zipfile
from email.parser import Parser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def check_wheel_contents(wheel: Path) -> str:
    """Check the renamed distribution while preserving import and CLI compatibility."""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "multi_modal_ai_studio/config/presets/speaches-jetson.yaml" in names
        assert "multi_modal_ai_studio/webui/static/index.html" in names
        assert not any(name.endswith(".backup") for name in names)
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_paths) == 1
        metadata_path = metadata_paths[0]
        metadata = Parser().parsestr(archive.read(metadata_path).decode("utf-8"))
        assert metadata["Name"] == "voice-ai-workbench", "Wrong public distribution name"
        assert metadata["Version"], "Missing package version"
        entries = configparser.ConfigParser()
        entries.read_string(
            archive.read(metadata_path.rsplit("/", 1)[0] + "/entry_points.txt").decode("utf-8")
        )
        for command in ("voice-ai-workbench", "multi-modal-ai-studio"):
            assert entries["console_scripts"].get(command) == "multi_modal_ai_studio.cli.main:main"
        html = archive.read("multi_modal_ai_studio/webui/static/index.html").decode("utf-8")
        assert "<title>Voice AI Workbench</title>" in html
        return metadata["Version"]


def check_distribution(wheel: Path) -> None:
    version = check_wheel_contents(wheel)
    env = {
        key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")
    }
    with tempfile.TemporaryDirectory(prefix="mmas-wheel-smoke-") as temporary:
        root = Path(temporary)
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        binary_dir = root / "venv" / ("Scripts" if os.name == "nt" else "bin")
        python = binary_dir / ("python.exe" if os.name == "nt" else "python")
        suffix = ".exe" if os.name == "nt" else ""
        cli = binary_dir / ("voice-ai-workbench" + suffix)
        subprocess.run(
            [str(python), "-m", "pip", "install", str(wheel)],
            cwd=root,
            env=env,
            check=True,
            timeout=600,
        )
        for command in ("voice-ai-workbench", "multi-modal-ai-studio"):
            executable = str(binary_dir / (command + suffix))
            result = subprocess.run(
                [executable, "--version"],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            assert version in result.stdout, result.stdout
            help_result = subprocess.run(
                [executable, "--help"],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            assert "Voice AI Workbench" in help_result.stdout
            print(result.stdout.strip())
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                (
                    "import multi_modal_ai_studio as m; print(m.__file__); "
                    "from pathlib import Path; import sys; import importlib.metadata as md; "
                    "assert md.version('voice-ai-workbench') == m.__version__; "
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
                        content = response.read()
                        assert content
                        if route == "/":
                            assert b"<title>Voice AI Workbench</title>" in content
                print("PASS: renamed wheel, both CLIs, legacy import, preset API, and branded UI")
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
