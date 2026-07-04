"""Import the top-level package so its __init__ metadata is covered."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def test_package_exposes_version():
    import src

    assert isinstance(src.__version__, str)
    assert src.__version__
