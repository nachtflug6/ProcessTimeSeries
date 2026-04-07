"""Compatibility test runner.

Prefer running the suite directly with:
    python3 -m pytest
"""

import importlib.util
import subprocess
import sys
import unittest


if __name__ == "__main__":
    if importlib.util.find_spec("pytest") is not None:
        raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "tests"]))

    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir="tests", pattern="test_*.py")
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)

