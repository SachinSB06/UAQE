#!/usr/bin/env python3
"""Pytest compatibility runner for UAQE test suites.

Translates pytest CLI invocation flags (-q, -v, -vv, -x, file::test)
to unittest test suites when pytest package is not installed in the environment.
"""

import os
import sys
import argparse
import unittest
import importlib.util
from pathlib import Path

# Ensure src and project root are in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(1, PROJECT_ROOT)


def load_tests_from_spec(spec_str: str) -> unittest.TestSuite:
    """Load tests from a path or path::TestName specification."""
    suite = unittest.TestSuite()
    
    if "::" in spec_str:
        parts = spec_str.split("::")
        file_path = os.path.abspath(parts[0])
        target_name = parts[-1]
        
        # Load module from file_path
        mod_name = Path(file_path).stem
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        
        # Search for TestCase classes in module
        found = False
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and issubclass(attr, unittest.TestCase):
                if hasattr(attr, target_name):
                    suite.addTest(attr(target_name))
                    found = True
                elif len(parts) >= 3 and attr.__name__ == parts[1] and hasattr(attr, target_name):
                    suite.addTest(attr(target_name))
                    found = True
        if not found:
            # Fallback: maybe target_name is a TestCase class
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if isinstance(attr, type) and issubclass(attr, unittest.TestCase):
                    if attr.__name__ == target_name:
                        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(attr))
                        found = True
        return suite

    # Regular file or directory path
    path = os.path.abspath(spec_str)
    if os.path.isfile(path):
        mod_name = Path(path).stem
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(mod))
    elif os.path.isdir(path):
        suite.addTests(unittest.defaultTestLoader.discover(path))
    else:
        # Try finding file under tests/ or src/uaqe/tests/
        cand1 = os.path.join(PROJECT_ROOT, "tests", spec_str)
        cand2 = os.path.join(SRC_DIR, "uaqe", "tests", spec_str)
        if os.path.exists(cand1):
            return load_tests_from_spec(cand1)
        elif os.path.exists(cand2):
            return load_tests_from_spec(cand2)
        else:
            raise FileNotFoundError(f"Path or test not found: {spec_str}")
            
    return suite


def main():
    args = sys.argv[1:]
    verbosity = 1
    failfast = False
    targets = []
    
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg in ("-q", "--quiet"):
            verbosity = 0
        elif arg == "-v":
            verbosity = 2
        elif arg in ("-vv", "-vvv"):
            verbosity = 2
        elif arg in ("-x", "--exitfirst"):
            failfast = True
        elif arg.startswith("-"):
            # Ignore other flags like -s, -r, etc.
            pass
        else:
            targets.append(arg)
        idx += 1

    overall_suite = unittest.TestSuite()
    if targets:
        for t in targets:
            try:
                sub_suite = load_tests_from_spec(t)
                overall_suite.addTests(sub_suite)
            except Exception as e:
                print(f"Error loading test {t}: {e}", file=sys.stderr)
                sys.exit(1)
    else:
        # Default discovery: tests/ and src/uaqe/tests/
        tests_dir = os.path.join(PROJECT_ROOT, "tests")
        uaqe_tests_dir = os.path.join(SRC_DIR, "uaqe", "tests")
        if os.path.exists(tests_dir):
            overall_suite.addTests(unittest.defaultTestLoader.discover(tests_dir))
        if os.path.exists(uaqe_tests_dir):
            overall_suite.addTests(unittest.defaultTestLoader.discover(uaqe_tests_dir))

    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    result = runner.run(overall_suite)
    
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
