#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# set -x # Echo commands before executing

# Define directories
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
PROTOC_DIR="${BASE_DIR}/proto/compiler"
PROTOC_BIN="${PROTOC_DIR}/bin/protoc"
# New output directory structure for packaging
PYTHON_OUT_PARENT_DIR="${BASE_DIR}/proto/generated"
PYTHON_PKG_NAME="pb_stubs"
PYTHON_OUT_DIR="${PYTHON_OUT_PARENT_DIR}/${PYTHON_PKG_NAME}"

PROTO_SRC_DIR="${BASE_DIR}/proto/src"
CUSTOM_PLUGIN_SCRIPT="proto/scripts/protoc-gen-custom-wrapper.sh"
CUSTOM_PLUGIN_BIN="${BASE_DIR}/${CUSTOM_PLUGIN_SCRIPT}"

# Check if protoc exists
if [ ! -f "${PROTOC_BIN}" ]; then
    echo "Error: protoc not found at ${PROTOC_BIN}"
    echo "Please ensure protoc is installed and placed in ${PROTOC_DIR}/bin/"
    exit 1
fi

# Check if custom plugin exists and is executable
if [ ! -f "${CUSTOM_PLUGIN_BIN}" ]; then
    echo "Error: Custom plugin not found at ${CUSTOM_PLUGIN_BIN}"
    exit 1
fi
chmod +x "${CUSTOM_PLUGIN_BIN}"

# Clean and recreate the Python output directory
echo "Cleaning and recreating Python output directory: ${PYTHON_OUT_DIR}"
rm -rf "${PYTHON_OUT_DIR}"
mkdir -p "${PYTHON_OUT_DIR}"

echo "Generating Python stubs..."

PROTO_INCLUDE_PATHS=(
    "-I${PROTO_SRC_DIR}/cosmos_sdk"
    "-I${PROTO_SRC_DIR}/mayanode"
    "-I${PROTO_SRC_DIR}"
    "-I${PROTOC_DIR}/include"
)

sdk_protos_to_compile=()
COSMOS_SDK_SRC_DIR="${PROTO_SRC_DIR}/cosmos_sdk"
if [ -d "${COSMOS_SDK_SRC_DIR}" ]; then
    echo "Searching for .proto files in ${COSMOS_SDK_SRC_DIR}..."
    while IFS= read -r -d $'\0' file; do
        relative_file="${file#${COSMOS_SDK_SRC_DIR}/}"
        sdk_protos_to_compile+=("$relative_file")
    done < <(find "${COSMOS_SDK_SRC_DIR}" -type f -name '*.proto' -print0)
    echo "Found ${#sdk_protos_to_compile[@]} .proto files in ${COSMOS_SDK_SRC_DIR}."
fi

mayanode_protos_to_compile=()
MAYANODE_SRC_DIR="${PROTO_SRC_DIR}/mayanode"
if [ -d "${MAYANODE_SRC_DIR}" ]; then
    echo "Searching for .proto files in ${MAYANODE_SRC_DIR}..."
    while IFS= read -r -d $'\0' file; do
        relative_file="${file#${MAYANODE_SRC_DIR}/}"
        mayanode_protos_to_compile+=("$relative_file")
    done < <(find "${MAYANODE_SRC_DIR}" -type f -name '*.proto' -print0)
    echo "Found ${#mayanode_protos_to_compile[@]} .proto files in ${MAYANODE_SRC_DIR}."
fi

other_protos_to_compile=()
echo "Searching for .proto files in ${PROTO_SRC_DIR} (excluding ${COSMOS_SDK_SRC_DIR} and ${MAYANODE_SRC_DIR})..."
while IFS= read -r -d $'\0' file; do
    relative_file="${file#${PROTO_SRC_DIR}/}"
    other_protos_to_compile+=("$relative_file")
done < <(find "${PROTO_SRC_DIR}" \( -path "${COSMOS_SDK_SRC_DIR}" -o -path "${MAYANODE_SRC_DIR}" \) -prune -o -type f -name '*.proto' -print0)
echo "Found ${#other_protos_to_compile[@]} .proto files in ${PROTO_SRC_DIR} (excluding cosmos_sdk and mayanode)."

all_found_protos_combined=("${sdk_protos_to_compile[@]}" "${mayanode_protos_to_compile[@]}" "${other_protos_to_compile[@]}")
unique_protos_to_compile=()
if [ ${#all_found_protos_combined[@]} -gt 0 ]; then
    unique_protos_to_compile=($(printf "%s\n" "${all_found_protos_combined[@]}" | grep -Ev '^[[:space:]]*$' | sort -u | tr '\n' ' '))
fi

if [ ${#unique_protos_to_compile[@]} -eq 0 ]; then
    echo "No .proto files found to compile. Please check paths and find commands."
    exit 1
fi

echo "Total unique .proto files to be compiled: (${#unique_protos_to_compile[@]} files)"

"${PROTOC_BIN}" \
    "${PROTO_INCLUDE_PATHS[@]}" \
    --plugin=protoc-gen-custom="${CUSTOM_PLUGIN_BIN}" \
    --custom_out="${PYTHON_OUT_DIR}" \
    --custom_opt="root_package=${PYTHON_PKG_NAME}" \
    "${unique_protos_to_compile[@]}"

# echo "Patching relative imports in generated Python files..." # Removed sed patch
# find "${PYTHON_OUT_DIR}" -name '*.py' -type f -print0 | while IFS= read -r -d $'\0' py_file; do
#     tmp_file=$(mktemp)
#     sed -E 's/^from (\.+)(cometbft|cosmos|google|gogoproto|kuji|mayachain|stargateexample|tendermint|types|amino|ibc|cosmos_proto)(.*)$/from \2\3/g' "${py_file}" > "${tmp_file}" && mv "${tmp_file}" "${py_file}"
# done
# echo "Finished patching relative imports."

find "${PYTHON_OUT_DIR}" -type d -exec touch {}/__init__.py \;
# Also ensure the parent generated directory is a package if we are importing pb_stubs from its parent
# touch "${PYTHON_OUT_PARENT_DIR}/__init__.py" # Not strictly necessary if PYTHON_OUT_PARENT_DIR itself is in sys.path

echo "Python stubs generated successfully in ${PYTHON_OUT_DIR}"
echo "Please ensure that ${PYTHON_OUT_PARENT_DIR} is in your PYTHONPATH or sys.path for 'import ${PYTHON_PKG_NAME}'"

# set +x # Stop echoing commands

# Reminder: If you encounter 'module not found' errors for generated files,
# ensure that the PYTHON_OUT_DIR (or its relevant subdirectories) is in your Python path.
# For example, you might need to add proto/generated/temp_python_out to PYTHONPATH.
# Or, in Python:
# import sys
# sys.path.insert(0, 'path/to/your/proto/generated/temp_python_out') 