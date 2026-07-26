#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/pxy"

echo "=== Building for macOS (native) ==="
zig build
mv zig-out/bin/pxy ../../pxy

echo "=== Building for Linux aarch64 ==="
zig build -Dtarget=aarch64-linux-gnu
mv zig-out/bin/pxy ../../pxy-linux

echo "=== Done ==="
ls -lh ../../pxy ../../pxy-linux
