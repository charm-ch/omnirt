#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN=python3.11
  else
    PYTHON_BIN=python
  fi
fi

if [[ -z "${MKDOCS_BIN:-}" ]]; then
  if command -v mkdocs >/dev/null 2>&1; then
    MKDOCS_BIN=mkdocs
  else
    MKDOCS_BIN="${PYTHON_BIN} -m mkdocs"
  fi
fi

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("OmniRT release gate requires Python >= 3.10; set PYTHON_BIN=/path/to/python3.11")
PY

"${PYTHON_BIN}" scripts/generate_models_doc.py --check
"${PYTHON_BIN}" scripts/check_bilingual_parity.py
"${PYTHON_BIN}" -m pytest tests/unit tests/parity --maxfail=1 -q
${MKDOCS_BIN} build --strict --clean
