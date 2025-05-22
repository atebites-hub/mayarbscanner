#!/bin/bash
# Wrapper script to execute betterproto plugin via python -m
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"
 
exec "$PYTHON_EXEC" -m betterproto.plugin "$@" 