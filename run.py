"""Quick launcher — starts the Face Tracking API and opens the demo UI."""
import os
import subprocess
import sys
import threading
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(ROOT, ".venv", "bin", "python3")
PYTHON = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

VENV_DIR = os.path.join(ROOT, ".venv")

# Re-launch under the venv interpreter if we're not already inside it.
if os.path.exists(VENV_PYTHON) and not sys.prefix.startswith(VENV_DIR):
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

import uvicorn  # noqa: E402 — only reachable once we're inside the venv

URL = "http://localhost:8000"


def _open_browser():
    time.sleep(2)
    webbrowser.open(URL)


threading.Thread(target=_open_browser, daemon=True).start()
uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
