# Codebase Reorganization Plan

## Current Issues

1. **Flat structure** - All Python modules in root directory
2. **Mixed concerns** - Code, data, docs, and generated files all mixed together
3. **No package structure** - Hard to scale or add new features
4. **Generated files in root** - Charts, PDFs, CSVs scattered
5. **No tests directory** - Test file mixed with source code
6. **Documentation scattered** - Multiple .md files in root

## Proposed Structure

```
et-intel/
├── README.md                    # Main entry point
├── requirements.txt
├── setup.py                     # Package installation (optional)
├── .gitignore                  # Exclude generated files
│
├── et_intel/                    # Main package
│   ├── __init__.py             # Package exports
│   ├── config.py               # Configuration
│   │
│   ├── core/                   # Core processing modules
│   │   ├── __init__.py
│   │   ├── pipeline.py         # Main orchestrator
│   │   ├── ingestion.py        # CSV import
│   │   ├── entity_extraction.py
│   │   └── sentiment_analysis.py
│   │
│   ├── reporting/              # Report generation
│   │   ├── __init__.py
│   │   └── report_generator.py
│   │
│   └── cli/                    # Command-line interfaces
│       ├── __init__.py
│       └── cli.py
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_system.py          # End-to-end tests
│   ├── test_ingestion.py
│   ├── test_entities.py
│   └── test_sentiment.py
│
├── data/                        # Data directory (gitignored)
│   ├── sample/                 # Sample/test data
│   │   └── sample_data.csv
│   ├── uploads/                # Raw CSV uploads
│   ├── processed/              # Processed comments
│   ├── database/               # Entity knowledge base
│   │   └── known_entities.json
│   └── reports/                # Generated reports
│       ├── pdfs/
│       └── charts/
│
├── docs/                        # Documentation
│   ├── INDEX.md
│   ├── PROJECT_SUMMARY.md
│   ├── QUICKSTART.md
│   ├── LAUNCH_PLAN.md
│   ├── CHANGELOG.md
│   └── API_REFERENCE.md        # Future: API docs
│
├── scripts/                     # Utility scripts
│   ├── migrate_data.py         # Migration helpers
│   └── cleanup.py              # Cleanup old reports
│
└── .env.example                # Environment template
```

## Benefits

### 1. **Clear Separation of Concerns**
- Source code in `et_intel/` package
- Tests isolated in `tests/`
- Documentation in `docs/`
- Data files in `data/` (gitignored)

### 2. **Scalability**
- Easy to add new modules (e.g., `et_intel/api/`, `et_intel/dashboard/`)
- Clear boundaries between components
- Can split into sub-packages as needed

### 3. **Professional Structure**
- Follows Python packaging conventions
- Can be installed as package: `pip install -e .`
- Better for CI/CD and deployment

### 4. **Maintainability**
- Related code grouped together
- Easy to find files
- Clear import paths: `from et_intel.core import pipeline`

### 5. **Testing**
- Dedicated tests directory
- Can add unit tests per module
- Test data separated from source

## Migration Strategy

### Phase 1: Create New Structure (Non-Breaking)
1. Create new directories
2. Move files to new locations
3. Update imports in moved files
4. Add `__init__.py` files with exports
5. Keep old files temporarily with deprecation warnings

### Phase 2: Update Entry Points
1. Update `cli.py` to use new imports
2. Update `test_system.py` to use new imports
3. Update documentation with new paths

### Phase 3: Cleanup
1. Remove old files
2. Update `.gitignore`
3. Update all documentation

## Import Changes

### Before:
```python
import config
from ingestion import CommentIngester
from pipeline import ETIntelligencePipeline
```

### After:
```python
from et_intel import config
from et_intel.core import CommentIngester, ETIntelligencePipeline
```

Or with package-level exports in `__init__.py`:
```python
from et_intel import CommentIngester, ETIntelligencePipeline
```

## Backward Compatibility

To maintain backward compatibility during transition:

1. **Keep old files with redirects**:
```python
# Old location: pipeline.py
from et_intel.core.pipeline import *
import warnings
warnings.warn("Import from et_intel.core.pipeline instead", DeprecationWarning)
```

2. **Or use symlinks** (if on Unix)

3. **Or update all references at once** (cleaner but breaking)

## Configuration Updates Needed

1. **config.py** - Update paths:
```python
BASE_DIR = Path(__file__).parent.parent  # Go up from et_intel/
DATA_DIR = BASE_DIR / "data"
```

2. **All modules** - Update imports to use package structure

3. **CLI** - Update to use package imports

4. **Tests** - Update to use package imports

## File Moves

| Current Location | New Location |
|----------------|--------------|
| `config.py` | `et_intel/config.py` |
| `pipeline.py` | `et_intel/core/pipeline.py` |
| `ingestion.py` | `et_intel/core/ingestion.py` |
| `entity_extraction.py` | `et_intel/core/entity_extraction.py` |
| `sentiment_analysis.py` | `et_intel/core/sentiment_analysis.py` |
| `report_generator.py` | `et_intel/reporting/report_generator.py` |
| `cli.py` | `et_intel/cli/cli.py` |
| `test_system.py` | `tests/test_system.py` |
| `sample_data.csv` | `data/sample/sample_data.csv` |
| All `.md` files | `docs/` |

## Next Steps

1. Review this plan
2. Create new directory structure
3. Move files systematically
4. Update all imports
5. Test thoroughly
6. Update documentation
7. Commit changes

## Alternative: Minimal Reorganization

If full reorganization is too disruptive, minimal changes:

```
et-intel/
├── src/
│   └── et_intel/          # All Python modules here
├── tests/
├── data/
├── docs/
└── scripts/
```

This keeps backward compatibility easier but is less clean.



