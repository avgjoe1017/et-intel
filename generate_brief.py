#!/usr/bin/env python3
"""Generate intelligence brief and open PDF"""

import sys
from pathlib import Path
import os

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from et_intel.core.pipeline import ETIntelligencePipeline
from et_intel.reporting.report_generator import IntelligenceBriefGenerator

print("\n" + "="*70)
print("  GENERATING INTELLIGENCE BRIEF")
print("="*70 + "\n")

# Initialize pipeline
pipeline = ETIntelligencePipeline(use_api=False)

# Generate brief
print("Generating intelligence brief from processed data...")
brief = pipeline.generate_intelligence_brief()

if not brief:
    print("\n[ERROR] No processed data found. Please process some CSV files first:")
    print("  python -m et_intel.cli.cli --batch")
    sys.exit(1)

print("\n[OK] Intelligence brief generated successfully!")

# Generate PDF
print("\nGenerating PDF report...")
generator = IntelligenceBriefGenerator()
pdf_path = generator.generate_report(brief)

print(f"\n[SUCCESS] PDF report generated!")
print(f"Location: {pdf_path}")
print(f"File size: {pdf_path.stat().st_size / 1024:.1f} KB")

# Try to open the PDF
if sys.platform == 'win32':
    os.startfile(str(pdf_path))
elif sys.platform == 'darwin':
    os.system(f'open "{pdf_path}"')
else:
    os.system(f'xdg-open "{pdf_path}"')

print("\n" + "="*70)
print("  BRIEF GENERATED SUCCESSFULLY")
print("="*70 + "\n")

