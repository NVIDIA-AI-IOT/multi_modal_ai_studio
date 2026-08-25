# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session hardware identity persistence tests."""

from multi_modal_ai_studio.config.schema import SessionConfig
from multi_modal_ai_studio.core.session import Session


def test_session_system_info_round_trip(tmp_path):
    session = Session(SessionConfig(), session_id="system-info-session")
    session.system_info = {
        "hostname": "jetson-test",
        "ip_addresses": ["10.0.0.5"],
        "device_model": "NVIDIA Jetson Test",
        "cpu": {"architecture": "aarch64", "logical_cores": 6},
        "gpu": {"name": "Orin (nvgpu)", "compute_capability": "8.7"},
        "memory_total_bytes": 8 * 1024**3,
    }
    path = tmp_path / "session.json"

    session.save(path)
    loaded = Session.load(path)

    assert loaded.system_info == session.system_info
    assert loaded.to_dict()["system_info"] == session.system_info
