#!/bin/bash
set -e

cd "$(dirname "$0")/.."

DEBUG=false
[ "$1" = "--debug" ] && DEBUG=true

WATCHOS="sim"
[ "$1" = "--release" ] && WATCHOS="device"

FLAG="-Dwatchos=$WATCHOS"
$DEBUG && FLAG="$FLAG -Ddebug-logs=true"

zig build $FLAG

cp outputs/lib/visual-watchos.a ../pixy\ Watch\ App/libvisual.a
echo "Deployed visual-watchos.a to pixy Watch App/libvisual.a"

SWIFT_TEST=$(mktemp /tmp/convnet-test-XXXXXX)
trap 'rm -f "$SWIFT_TEST"' EXIT

swiftc -enable-experimental-feature Extern -o "$SWIFT_TEST" scripts/test_convnet.swift outputs/lib/visual.a -framework CoreGraphics -framework ImageIO 2>&1

"$SWIFT_TEST"
