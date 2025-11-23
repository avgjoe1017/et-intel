#!/usr/bin/env python3
"""
Convenience script to run the CLI
This allows: python run_cli.py instead of python -m et_intel.cli.cli
"""

if __name__ == "__main__":
    from et_intel.cli.cli import main
    main()



