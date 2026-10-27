import os
import shutil
import subprocess
from pathlib import Path


def test_nspect_requirements_matches_uv_lock() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    generator = repo_root / "scripts" / "generate_nspect_requirements.sh"
    uv_bin = shutil.which("uv")

    assert uv_bin is not None, "uv is required by the dev extra"

    result = subprocess.run(
        [str(generator), "--check"],
        cwd=repo_root,
        env={**os.environ, "UV_BIN": uv_bin},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
