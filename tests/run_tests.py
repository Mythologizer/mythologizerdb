#!/usr/bin/env python3
"""
Test runner script for mythologizer_postgres tests.
"""

import sys
import subprocess
import argparse


def run_tests(test_type="all", verbose=False):
    """Run tests based on the specified type."""
    # Use uv to run pytest in the virtual environment
    cmd = ["uv", "run", "--env-file", ".env.test", "pytest", "tests/"]
    
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "all":
        pass  # Run all tests
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    if verbose:
        cmd.append("-v")
    
    # Add coverage if available
    try:
        cmd.extend(["--cov=mythologizer_postgres", "--cov-report=term-missing"])
    except ImportError:
        pass  # Coverage not available
    
    print(f"Running tests: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run mythologizer_postgres tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration"], 
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    success = run_tests(args.type, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 