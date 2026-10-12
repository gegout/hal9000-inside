#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
    PIP=".venv/bin/pip"
else
    PYTHON="python"
    PIP="pip"
fi

echo "=== 1. Checking package installation ==="
$PIP install -e . --quiet

echo "=== 2. Checking imports ==="
$PYTHON -c "import hal9000_inside; import hal9000_inside.assistant; import hal9000_inside.cli.main; import hal9000_inside.memory; import hal9000_inside.llm; import hal9000_inside.audio; import hal9000_inside.hardware; import hal9000_inside.mcp.server; print('All core modules imported successfully.')"

echo "=== 3. Running unit test suite ==="
$PYTHON -m unittest discover tests

echo "=== All checks passed successfully. ==="
