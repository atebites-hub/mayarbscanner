#!/bin/bash

# Exit on error
set -e

# Project root directory (assuming script is in proto/scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
PROTO_DIR="$PROJECT_ROOT/proto"
COMPILER_DIR="$PROTO_DIR/compiler"
SRC_DIR="$PROTO_DIR/src"
GENERATED_DIR="$PROTO_DIR/generated/mayapb" # Output to a sub-package

# Ensure output directory exists and is clean
rm -rf "$GENERATED_DIR"
mkdir -p "$GENERATED_DIR"

# Protoc binary
PROTOC_BIN="$COMPILER_DIR/bin/protoc"

# Include paths
INCLUDE_PATHS=(
  "-I=$SRC_DIR/mayanode"
  "-I=$SRC_DIR/cosmos_sdk"
  "-I=$SRC_DIR/gogoproto"
  "-I=$COMPILER_DIR/include" # For google.protobuf well-known types
  "-I=$SRC_DIR" # General include for any other potential relative paths
  "-I=$SRC_DIR/cosmos_proto" # Explicitly add cosmos_proto
)

MAYANODE_PROTO_FILES=$(find "$SRC_DIR/mayanode" -name '*.proto' -print)

if [ -z "$MAYANODE_PROTO_FILES" ]; then
  echo "No .proto files found in $SRC_DIR/mayanode"
  exit 1
fi

echo "Found Mayanode .proto files:"
echo "$MAYANODE_PROTO_FILES"
echo ""

echo "Generating Python stubs with betterproto..."
(cd "$PROJECT_ROOT" && \
  "$PROTOC_BIN" \
  "${INCLUDE_PATHS[@]}" \
  --plugin=protoc-gen-custom="$PROTO_DIR/scripts/protoc-gen-custom-wrapper.sh" \
  --custom_out="./proto/generated/mayapb" \
  $MAYANODE_PROTO_FILES \
)

# Create __init__.py files to make the generated code a package
touch "$PROTO_DIR/generated/__init__.py"
touch "$GENERATED_DIR/__init__.py"

echo ""
echo "Protobuf Python stubs generated successfully in $GENERATED_DIR"
echo "Make sure 'proto' is in your PYTHONPATH or handle imports appropriately."

# Make the script executable
chmod +x "$PROTO_DIR/scripts/generate_protos.sh" 