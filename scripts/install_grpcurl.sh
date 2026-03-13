#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install grpcurl from GitHub releases (linux arm64).
# Usage: ./scripts/install_grpcurl.sh [install-dir]
# Default install-dir: $HOME/.local/bin (ensure it's in your PATH)

set -e
VERSION="1.9.3"
INSTALL_DIR="${1:-$HOME/.local/bin}"
ARCH="linux_arm64"
TAR="grpcurl_${VERSION}_${ARCH}.tar.gz"
URL="https://github.com/fullstorydev/grpcurl/releases/download/v${VERSION}/${TAR}"

mkdir -p "$INSTALL_DIR"
echo "Downloading grpcurl v${VERSION} (${ARCH})..."
curl -sSL -o "/tmp/${TAR}" "$URL"
echo "Extracting to ${INSTALL_DIR}..."
tar -xzf "/tmp/${TAR}" -C "$INSTALL_DIR" grpcurl
rm -f "/tmp/${TAR}"
chmod +x "${INSTALL_DIR}/grpcurl"
echo "Installed: ${INSTALL_DIR}/grpcurl"
if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
  echo "Add to PATH: export PATH=\"${INSTALL_DIR}:\$PATH\""
  echo "Or add the above line to your ~/.bashrc"
fi
"${INSTALL_DIR}/grpcurl" -version
