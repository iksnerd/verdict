"""Documented import path, kept: the module is `verdict.backends.jev`."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("verdict.backends.jev")
