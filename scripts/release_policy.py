# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Decide whether a verified release build may enter the PyPI approval job."""

import os
import tomllib
from pathlib import Path

from packaging.version import Version


def publish_pypi(
    *, event: str, version: str, tag: str, prerelease: bool, destination: str, enabled: bool
) -> bool:
    """Keep PR/push builds and RC releases out of automatic index publishing.

    Explicit manual promotion may publish an RC. The workflow additionally
    verifies the main workflow ref, published release, and checked-out tag.
    Environment approval and Trusted Publishing are separate controls.
    """
    if event != "release" and not (event == "workflow_dispatch" and destination == "pypi"):
        return False
    package_version = Version(version)
    if not tag.startswith("v") or Version(tag[1:]) != package_version:
        raise ValueError(f"Release tag {tag!r} does not match package version {version!r}")
    if not enabled:
        return False
    if event == "workflow_dispatch":
        return True
    return (
        not prerelease and not package_version.is_prerelease and not package_version.is_devrelease
    )


if __name__ == "__main__":
    with Path("pyproject.toml").open("rb") as source:
        version = tomllib.load(source)["project"]["version"]
    decision = publish_pypi(
        event=os.environ["EVENT_NAME"],
        version=version,
        tag=os.environ["RELEASE_TAG"],
        prerelease=os.environ.get("RELEASE_PRERELEASE") == "true",
        destination=os.environ.get("DESTINATION", "none"),
        enabled=os.environ.get("PYPI_ENABLED") == "true",
    )
    print(f"publish_pypi={str(decision).lower()}")
