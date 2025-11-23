# Testing Guide

## Overview

The ET Social Intelligence System includes comprehensive end-to-end tests covering the complete pipeline from CSV ingestion to report generation.

## Test Structure

```
tests/
├── __init__.py           # Test package
├── conftest.py           # Pytest configuration and fixtures
├── test_system.py        # Original system test (legacy)
├── test_e2e.py          # Comprehensive end-to-end tests
├── test_dashboard.py     # Dashboard-specific tests
└── run_all_tests.py     # Test runner script
```

## Running Tests

### Quick Start
```bash
# Run all tests
python tests/run_all_tests.py

# Or use pytest directly
pytest tests/ -v
```

### Specific Test Suites
```bash
# End-to-end tests only
pytest tests/test_e2e.py -v

# Dashboard tests only
pytest tests/test_dashboard.py -v

# Original system test
python tests/test_system.py
```

### With Coverage
```bash
pip install pytest-cov
pytest tests/ --cov=et_intel --cov-report=html
```

## Test Coverage

### End-to-End Tests (`test_e2e.py`)

1. **Module Imports** - Verify all modules can be imported
2. **CSV Ingestion** - Test CSV file reading and standardization
3. **Entity Extraction** - Test entity detection (people, shows, couples)
4. **Sentiment Analysis (Rule-based)** - Test rule-based sentiment
5. **Sentiment Analysis (TextBlob)** - Test TextBlob baseline
6. **Full Pipeline** - Test complete processing pipeline
7. **Intelligence Brief** - Test brief generation
8. **PDF Report** - Test PDF generation
9. **Batch Processing** - Test batch CSV processing
10. **CSV Tracking** - Test processed CSV tracking
11. **Database Integration** - Test SQLite database
12. **Relationship Graphs** - Test NetworkX graph generation
13. **Velocity Calculation** - Test sentiment velocity
14. **Error Handling** - Test invalid input handling
15. **Configuration** - Test config loading

### Dashboard Tests (`test_dashboard.py`)

1. **Dashboard Imports** - Verify Streamlit/Plotly available
2. **Data Loading** - Test data loading from processed files
3. **Data Filtering** - Test platform and date filtering
4. **Metrics Calculation** - Test dashboard metrics
5. **Chart Data Preparation** - Test chart data aggregation
6. **Empty Data Handling** - Test graceful handling of no data

## Test Data

Tests use temporary test data created on-the-fly:
- Sample CSV files with realistic comment data
- Temporary directories for processed data
- Isolated test environment (cleaned up after tests)

## Writing New Tests

### Example Test
```python
def test_my_feature(self):
    """Test description"""
    # Setup
    ingester = CommentIngester()
    
    # Execute
    df = ingester.ingest_instagram_csv("test.csv")
    
    # Assert
    assert len(df) > 0
    assert 'comment_text' in df.columns
```

### Using Fixtures
```python
@pytest.fixture
def sample_data():
    """Create sample data"""
    return pd.DataFrame({...})

def test_with_fixture(sample_data):
    """Use fixture"""
    result = process(sample_data)
    assert result is not None
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python -m spacy download en_core_web_sm
      - run: pytest tests/ -v
```

## Troubleshooting

### Tests Fail with Import Errors
```bash
# Make sure you're in project root
cd /path/to/et-intel

# Install dependencies
pip install -r requirements.txt
```

### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
# Or for better accuracy:
python -m spacy download en_core_web_lg
```

### Database Tests Fail
- Tests use temporary databases
- Check write permissions in test directory
- SQLite is optional - tests will skip if unavailable

### Dashboard Tests Skip
- Dashboard tests require Streamlit/Plotly
- Install: `pip install streamlit plotly`
- Tests will skip gracefully if not available

## Test Best Practices

1. **Isolation**: Each test is independent
2. **Cleanup**: Tests clean up temporary files
3. **Fixtures**: Use fixtures for common setup
4. **Assertions**: Clear, specific assertions
5. **Error Messages**: Helpful error messages

## Performance

- **Fast Tests**: Most tests run in <1 second
- **Slow Tests**: Some tests (HF model loading) take longer
- **Parallel**: Run with `pytest -n auto` (requires pytest-xdist)

## Coverage Goals

- **Target**: 80%+ code coverage
- **Current**: Core modules well-tested
- **Focus Areas**: New features, edge cases

---

**Run tests regularly** to catch regressions early!

