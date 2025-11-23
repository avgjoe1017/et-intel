# Migration Guide - Reorganization v1.0.0

## Overview

The codebase has been reorganized into a proper Python package structure for better maintainability and scalability.

## What Changed

### Directory Structure

**Before:**
```
et-intel/
├── config.py
├── pipeline.py
├── ingestion.py
├── entity_extraction.py
├── sentiment_analysis.py
├── report_generator.py
├── cli.py
├── test_system.py
└── *.md files
```

**After:**
```
et-intel/
├── et_intel/              # Main package
│   ├── __init__.py
│   ├── config.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── ingestion.py
│   │   ├── entity_extraction.py
│   │   └── sentiment_analysis.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── report_generator.py
│   └── cli/
│       ├── __init__.py
│       └── cli.py
├── tests/
│   ├── __init__.py
│   └── test_system.py
├── data/
│   ├── sample/
│   ├── uploads/
│   ├── processed/
│   ├── database/
│   └── reports/
├── docs/                   # All documentation
└── scripts/               # Utility scripts
```

## Import Changes

### Old Imports
```python
import config
from pipeline import ETIntelligencePipeline
from ingestion import CommentIngester
from entity_extraction import EntityExtractor
from sentiment_analysis import SentimentAnalyzer
from report_generator import IntelligenceBriefGenerator
```

### New Imports
```python
# Option 1: Package-level imports (recommended)
from et_intel import config
from et_intel import ETIntelligencePipeline, IntelligenceBriefGenerator

# Option 2: Direct module imports
from et_intel.core import ETIntelligencePipeline
from et_intel.core.ingestion import CommentIngester
from et_intel.core.entity_extraction import EntityExtractor
from et_intel.core.sentiment_analysis import SentimentAnalyzer
from et_intel.reporting.report_generator import IntelligenceBriefGenerator
```

## CLI Usage Changes

### Old
```bash
python cli.py --import data.csv --platform instagram
python cli.py --generate --last-30-days
```

### New
```bash
# Option 1: As module
python -m et_intel.cli.cli --import data.csv --platform instagram
python -m et_intel.cli.cli --generate --last-30-days

# Option 2: After installing package (pip install -e .)
et-intel --import data.csv --platform instagram
et-intel --generate --last-30-days
```

## Path Changes

### Configuration Paths
All paths in `config.py` now reference the project root correctly:
- `data/` is now at project root level
- `reports/` moved to `data/reports/pdfs/`
- Charts moved to `data/reports/charts/`

### Data Files
- Sample data: `data/sample/sample_data.csv`
- Processed data: `data/processed/`
- Reports: `data/reports/pdfs/`
- Entity database: `data/database/known_entities.json`

## Testing

### Old
```bash
python test_system.py
```

### New
```bash
# From project root
python -m pytest tests/
# Or
python tests/test_system.py
```

## Installation

You can now install the package in development mode:

```bash
pip install -e .
```

This allows you to import `et_intel` from anywhere.

## Backward Compatibility

**Note:** The old flat structure files have been removed. If you have scripts using the old imports, you'll need to update them.

## Migration Steps

1. **Update your scripts:**
   - Change imports to use `et_intel` package
   - Update file paths to new locations

2. **Update CLI usage:**
   - Use `python -m et_intel.cli.cli` instead of `python cli.py`

3. **Move data files:**
   - If you have existing processed data, move it to `data/processed/`
   - Move reports to `data/reports/pdfs/`

4. **Test everything:**
   ```bash
   python tests/test_system.py
   ```

## Questions?

If you encounter issues during migration, check:
- `docs/README.md` - Updated documentation
- `docs/QUICKSTART.md` - Quick start guide
- `tests/test_system.py` - Example usage



