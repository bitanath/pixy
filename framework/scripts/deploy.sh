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

cp outputs/lib/llm-watchos.a ../pixy\ Watch\ App/libllm.a
echo "Deployed llm-watchos.a to pixy Watch App/libllm.a"

SWIFT_TEST=$(mktemp /tmp/llm-test-XXXXXX)
trap 'rm -f "$SWIFT_TEST"' EXIT

swiftc -enable-experimental-feature Extern -o "$SWIFT_TEST" scripts/test.swift outputs/lib/llm.a 2>&1

"$SWIFT_TEST"
