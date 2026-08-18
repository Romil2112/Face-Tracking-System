"""Quick launcher — starts the Face Tracking API and opens the docs."""
import threading
import time
import webbrowser

import uvicorn

URL = "http://localhost:8000"


def _open_browser():
    time.sleep(2)
    webbrowser.open(URL)


threading.Thread(target=_open_browser, daemon=True).start()
uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
