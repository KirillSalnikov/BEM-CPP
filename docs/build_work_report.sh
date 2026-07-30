#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 docs/build_work_report_assets.py
weasyprint docs/work_report_ru.html docs/work_report_ru.pdf
printf 'Built %s\n' "$ROOT/docs/work_report_ru.pdf"
