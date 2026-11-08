#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/../.cache}"

# Il Markdown è la sorgente editoriale; il frammento non contiene un preambolo.
pandoc sezione_sperimentale.md --from=markdown --to=latex \
  --top-level-division=section --no-highlight --wrap=preserve \
  --output=sezione_sperimentale.tex
python3 - <<'PY'
from pathlib import Path
import re
p = Path('sezione_sperimentale.tex')
text = p.read_text(encoding='utf-8')
# Mantiene il frammento compatibile anche con pdfLaTeX, usato nella teoria.
text = text.replace('−', r'\ensuremath{-}')
text = re.sub(r'\\\[(.*?)\\\]',
              lambda m: '\\begin{equation}' + m.group(1) + '\\end{equation}',
              text, flags=re.S)
# I percorsi lunghi possono spezzarsi sugli underscore e sui separatori.
def readable_code(match):
    value = match.group(1)
    is_path = any(x in value for x in ('.', '/')) or value.startswith('results\\_')
    if is_path and not any(x in value for x in ('{', '}', '\\/-')):
        value = re.sub(r'\\([_#$%&])', r'\1', value)
        return '\\nolinkurl{' + value + '}'
    return match.group(0)
text = re.sub(r'\\texttt\{([^{}]*)\}', readable_code, text)
text = text.replace('\\begin{longtable}', '\\begingroup\\small\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}')
text = text.replace('\\end{longtable}', '\\end{longtable}\n\\endgroup')
text = text.replace('\\begin{verbatim}', '\\begingroup\\footnotesize\n\\begin{verbatim}')
text = text.replace('\\end{verbatim}', '\\end{verbatim}\n\\endgroup')
p.write_text('% Generato da sezione_sperimentale.md con compila.sh.\n' + text, encoding='utf-8')
# Un singolo file completo è più semplice da aprire in Overleaf.
wrapper = Path('anteprima.tex').read_text(encoding='utf-8')
standalone = wrapper.replace(r'\input{sezione_sperimentale.tex}', text)
Path('sezione_sperimentale_completa.tex').write_text(
    '% !TeX program = xelatex\n'
    '% Documento autonomo generato con compila.sh; compilare con XeLaTeX.\n'
    + standalone, encoding='utf-8')
PY

xelatex -interaction=nonstopmode -halt-on-error \
  -jobname=sezione_sperimentale sezione_sperimentale_completa.tex > compilazione.log
xelatex -interaction=nonstopmode -halt-on-error \
  -jobname=sezione_sperimentale sezione_sperimentale_completa.tex >> compilazione.log
python3 - <<'PY'
from zipfile import ZIP_DEFLATED, ZipFile
files = (
    'sezione_sperimentale.pdf', 'sezione_sperimentale_completa.tex',
    'sezione_sperimentale.tex', 'sezione_sperimentale.md',
    'README.md', 'anteprima.tex', 'compila.sh',
)
with ZipFile('sezione_sperimentale.zip', 'w', compression=ZIP_DEFLATED) as archive:
    for name in files:
        archive.write(name)
PY
printf '%s\n' 'Creati il frammento LaTeX, il documento LaTeX autonomo, il PDF e lo ZIP'
