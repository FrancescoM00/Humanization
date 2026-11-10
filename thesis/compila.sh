#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/../.cache}"

pandoc sezione_sperimentale.md --from=markdown --to=latex \
  --top-level-division=section --no-highlight --wrap=preserve \
  --output=sezione_sperimentale.tex
pandoc sintesi_incontro_professore.md --from=markdown --to=latex \
  --top-level-division=section --no-highlight --wrap=preserve \
  --output=sintesi_incontro_professore_contenuto.tex
python3 - <<'PY'
from pathlib import Path
import re

def convert(path):
    text = path.read_text(encoding='utf-8')
    text = text.replace('−', r'\ensuremath{-}')
    text = re.sub(r'\\\[(.*?)\\\]', lambda m: '\\begin{equation}' + m.group(1) + '\\end{equation}', text, flags=re.S)
    def readable_code(match):
        value = match.group(1)
        if any(x in value for x in ('.', '/')) and not any(x in value for x in ('{', '}', '\\/-')):
            value = re.sub(r'\\([_#$%&])', r'\1', value)
            return '\\nolinkurl{' + value + '}'
        return match.group(0)
    text = re.sub(r'\\texttt\{([^{}]*)\}', readable_code, text)
    text = text.replace('\\begin{longtable}', '\\begingroup\\small\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}')
    text = text.replace('\\end{longtable}', '\\end{longtable}\n\\endgroup')
    text = text.replace('\\begin{verbatim}', '\\begingroup\\footnotesize\n\\begin{verbatim}')
    text = text.replace('\\end{verbatim}', '\\end{verbatim}\n\\endgroup')
    path.write_text('% Generato dal Markdown con compila.sh.\n' + text, encoding='utf-8')
    return text

chapter = convert(Path('sezione_sperimentale.tex'))
brief = convert(Path('sintesi_incontro_professore_contenuto.tex'))
for wrapper, content, target, marker in [
    ('anteprima.tex', chapter, 'sezione_sperimentale_completa.tex', r'\input{sezione_sperimentale.tex}'),
    ('anteprima_sintesi.tex', brief, 'sintesi_incontro_professore.tex', r'\input{sintesi_incontro_professore_contenuto.tex}'),
]:
    text = Path(wrapper).read_text(encoding='utf-8').replace(marker, content)
    Path(target).write_text('% !TeX program = xelatex\n% Documento autonomo; compilare con XeLaTeX.\n' + text, encoding='utf-8')
PY

for pass in 1 2; do
  xelatex -interaction=nonstopmode -halt-on-error \
    -jobname=sezione_sperimentale sezione_sperimentale_completa.tex > "compilazione_${pass}.log"
  xelatex -interaction=nonstopmode -halt-on-error \
    -jobname=sintesi_incontro_professore sintesi_incontro_professore.tex > "compilazione_sintesi_${pass}.log"
done
python3 - <<'PY'
from zipfile import ZIP_DEFLATED, ZipFile
chapter = ['sezione_sperimentale.pdf', 'sezione_sperimentale_completa.tex', 'sezione_sperimentale.tex',
           'sezione_sperimentale.md', 'README.md', 'anteprima.tex', 'compila.sh',
           'sintesi_incontro_professore.md', 'sintesi_incontro_professore.pdf',
           'sintesi_incontro_professore.tex', 'sintesi_incontro_professore_contenuto.tex', 'anteprima_sintesi.tex']
with ZipFile('sezione_sperimentale.zip', 'w', compression=ZIP_DEFLATED) as archive:
    for name in chapter:
        archive.write(name)
with ZipFile('sintesi_incontro_professore.zip', 'w', compression=ZIP_DEFLATED) as archive:
    for name in ['sintesi_incontro_professore.pdf', 'sintesi_incontro_professore.tex', 'sintesi_incontro_professore.md']:
        archive.write(name)
print('Creati PDF, LaTeX e ZIP della sezione sperimentale e della sintesi.')
PY
