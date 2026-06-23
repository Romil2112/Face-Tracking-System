"""Pytest configuration: put the application source on the import path.

The modules under ``src/`` import each other by bare name (e.g. ``import config``),
so ``src/`` must be importable directly.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
