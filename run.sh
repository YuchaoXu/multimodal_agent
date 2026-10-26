#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${MMAGENT_PYTHON:-}" ]]; then
  exec "$MMAGENT_PYTHON" -m mmagent "$@"
fi
for candidate in .venv/bin/python python3.12 python3.11 python3.10 python3 "$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"; do
  if "$candidate" -c 'import sys; assert sys.version_info >= (3, 10); import PIL, pypdf, pypdfium2, numpy' >/dev/null 2>&1; then
    exec "$candidate" -m mmagent "$@"
  fi
done
printf '%s\n' 'No Python >=3.10 with dependencies found. See README installation, or set MMAGENT_PYTHON.' >&2
exit 1
