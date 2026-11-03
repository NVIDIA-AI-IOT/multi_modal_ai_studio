# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Copy the canonical repository presets into both wheel build paths."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithPresets(build_py):
    def run(self):
        super().run()
        source = Path(__file__).resolve().parent / "presets"
        destination = Path(self.build_lib) / "multi_modal_ai_studio" / "config" / "presets"
        self.copy_tree(str(source), str(destination))


setup(cmdclass={"build_py": BuildPyWithPresets})
