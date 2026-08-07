#!/usr/bin/env bash
set -euo pipefail

study_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$study_dir/../.." && pwd)"

cd "$repo_dir"
python3 scripts/analyze_convergence_study.py

cd "$study_dir"
if command -v tectonic >/dev/null 2>&1; then
  tectonic article_ru.tex
elif command -v xelatex >/dev/null 2>&1; then
  xelatex -interaction=nonstopmode -halt-on-error article_ru.tex
  xelatex -interaction=nonstopmode -halt-on-error article_ru.tex
elif [[ -x "$HOME/.local/bin/pdflatex" ]] &&
     "$HOME/.local/bin/pdflatex" --version 2>/dev/null | grep -qi tectonic; then
  "$HOME/.local/bin/pdflatex" article_ru.tex
  "$HOME/.local/bin/pdflatex" article_ru.tex
else
  printf '%s\n' \
    'A Unicode-capable TeX engine is required.' \
    'Install Tectonic or XeLaTeX, then rerun this script.' >&2
  exit 2
fi

printf 'Article: %s\n' "$study_dir/article_ru.pdf"
