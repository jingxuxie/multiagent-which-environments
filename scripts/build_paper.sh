#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../paper"
if command -v bibtex >/dev/null 2>&1; then BIB=bibtex
elif command -v bibtex.original >/dev/null 2>&1; then BIB=bibtex.original
else echo 'Install a TeX distribution providing bibtex.' >&2; exit 1; fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex
"$BIB" main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if grep -Eq 'undefined references|Citation .* undefined|Overfull' main.log; then
  echo 'Unresolved references or overflowing content; inspect paper/main.log.' >&2; exit 1
fi
