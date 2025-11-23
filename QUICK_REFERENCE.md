# Quick Reference Guide

## 🚀 Most Common Commands

### Process Data
```bash
# Batch process all unprocessed CSVs
python -m et_intel.cli.cli --batch

# Process single CSV
python -m et_intel.cli.cli --import file.csv --platform instagram --subject "Taylor Swift"
```

### Launch Dashboard
```bash
streamlit run streamlit_dashboard.py
```

### Run Tests
```bash
python tests/run_all_tests.py
```

### Generate Report
```bash
python -m et_intel.cli.cli --generate --last-30-days
```

---

## 📁 Key Directories

- `data/uploads/` - Place CSV files here
- `data/processed/` - Processed comment data
- `data/database/` - SQLite database and entity knowledge
- `data/reports/pdfs/` - Generated PDF reports
- `data/reports/charts/` - Chart images

---

## 🔑 Environment Variables

Create `.env` file:
```
OPENAI_API_KEY=your-key-here
```

---

## 📊 Dashboard Features

- **URL**: http://localhost:8501
- **Filters**: Platform, date range
- **Tabs**: Sentiment, Entities, Emotions, Relationships
- **Auto-refresh**: Every 5 minutes

---

## 🧪 Test Coverage

- **15 end-to-end tests** - Full pipeline
- **6 dashboard tests** - Dashboard functionality
- **Run**: `pytest tests/ -v`

---

## 🐛 Quick Fixes

**No data in dashboard?**
→ Process data first: `python -m et_intel.cli.cli --batch`

**Tests fail?**
→ Install dependencies: `pip install -r requirements.txt`

**spaCy model missing?**
→ `python -m spacy download en_core_web_lg`

**Dashboard port in use?**
→ `streamlit run streamlit_dashboard.py --server.port 8502`

---

## 📚 Documentation

- `README.md` - Main documentation
- `DASHBOARD_SETUP.md` - Dashboard guide
- `TESTING_GUIDE.md` - Testing documentation
- `QUICK_START_ENHANCED.md` - New features guide
- `ENHANCEMENTS_SUMMARY.md` - What's new

---

**Need help?** Check the docs or open an issue!

