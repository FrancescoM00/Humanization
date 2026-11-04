#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
# The managed cloud HTTP proxy is trusted; TLS and NLTK artifact checks stay enabled.
NLTK_ALLOW_PROXIED_URLOPEN=1 .venv/bin/python -m nltk.downloader -q -e -d "$PWD/.venv/nltk_data" cmudict
.venv/bin/python -m pip check
