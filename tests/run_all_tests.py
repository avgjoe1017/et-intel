#!/usr/bin/env python3
"""
Run all tests for ET Social Intelligence System
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run all tests"""
    project_root = Path(__file__).parent.parent
    
    print("\n" + "="*70)
    print("  ET SOCIAL INTELLIGENCE - TEST SUITE")
    print("="*70 + "\n")
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("❌ pytest not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
        import pytest
    
    # Run tests
    test_dir = Path(__file__).parent
    
    print("Running end-to-end tests...\n")
    result = pytest.main([
        str(test_dir),
        "-v",
        "--tb=short",
        "-W", "ignore::DeprecationWarning"
    ])
    
    print("\n" + "="*70)
    if result == 0:
        print("[PASS] All tests passed!")
    else:
        print(f"[FAIL] Some tests failed (exit code: {result})")
    print("="*70 + "\n")
    
    return result


if __name__ == "__main__":
    sys.exit(main())

