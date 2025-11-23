# ET SOCIAL INTELLIGENCE SYSTEM - FILE INDEX

## 📖 START HERE

If you're new, read these files IN THIS ORDER:

1. **PROJECT_SUMMARY.md** - What this is and why it matters (5 min read)
2. **QUICKSTART.md** - Get running in 5 minutes
3. **LAUNCH_PLAN.md** - Your next steps and timeline
4. **README.md** - Full technical documentation

## 📁 File Structure

### Documentation
- `INDEX.md` ← You are here
- `PROJECT_SUMMARY.md` - Executive overview and business case
- `QUICKSTART.md` - 5-minute setup guide
- `LAUNCH_PLAN.md` - Immediate action plan
- `README.md` - Complete technical documentation

### Core System Files
- `config.py` - All settings and configuration
- `cli.py` - Interactive command-line interface
- `pipeline.py` - Main orchestrator
- `ingestion.py` - CSV import engine
- `entity_extraction.py` - Celebrity/show detection
- `sentiment_analysis.py` - Emotion classification
- `report_generator.py` - PDF creation
- `test_system.py` - Verification script
- `requirements.txt` - Python dependencies

### Sample Data
- `sample_data.csv` - Test dataset (50 ET comments)

### Generated Outputs
- `data/processed/` - Analyzed comments with sentiment
- `data/database/` - Growing entity knowledge base
- `reports/` - Intelligence Brief PDFs
- `reports/TEST_Intelligence_Brief.pdf` - Sample report

## 🚀 Quick Commands

### Test the system
```bash
python3 test_system.py
```

### Run interactive interface
```bash
python3 cli.py
```

### Process a CSV file (Python)
```python
from pipeline import ETIntelligencePipeline

pipeline = ETIntelligencePipeline(use_api=False)
df, entities = pipeline.process_new_data(
    'comments.csv',
    'instagram',
    {'post_url': 'https://...', 'subject': 'Taylor Swift'}
)
```

### Generate intelligence brief
```python
brief = pipeline.generate_intelligence_brief()

from report_generator import IntelligenceBriefGenerator
generator = IntelligenceBriefGenerator()
pdf = generator.generate_report(brief)
```

## 📊 What Each File Does

### config.py
Central configuration. Edit this to:
- Set API keys
- Adjust velocity thresholds
- Add seed relationships
- Change batch sizes

### cli.py
User-friendly menu interface. Use this to:
- Import CSV files
- Generate reports
- View database stats
- Configure settings

### pipeline.py
Main brain. Coordinates:
- Data ingestion
- Entity extraction
- Sentiment analysis
- Report generation

### ingestion.py
CSV processor. Handles:
- Instagram comments
- YouTube comments
- Auto-column detection
- Deduplication

### entity_extraction.py
Smart detection engine. Finds:
- Celebrities mentioned
- Shows/movies discussed
- Couples (co-occurring entities)
- Storylines (lawsuits, relationships)

### sentiment_analysis.py
Emotion classifier. Provides:
- Context-aware sentiment
- 8 emotion categories
- Sarcasm detection
- Velocity tracking

### report_generator.py
PDF creator. Generates:
- Executive summary
- Velocity alerts
- Entity rankings
- Emotion charts
- Professional formatting

## 🎯 Common Tasks

### Import your first CSV
1. Run `python3 cli.py`
2. Select Option 1
3. Enter CSV path
4. Add metadata
5. Wait for processing

### Generate your first report
1. Run `python3 cli.py`
2. Select Option 2
3. Choose filters
4. Find PDF in `reports/`

### Add a new celebrity to track
Edit `data/database/known_entities.json`:
```json
{
  "people": [
    "Taylor Swift",
    "Your New Celebrity"
  ]
}
```

### Change velocity alert threshold
Edit `config.py`:
```python
VELOCITY_ALERT_THRESHOLD = 0.3  # 30% change
```

### Enable API mode for better accuracy
```bash
export OPENAI_API_KEY="your-key"
```
Then set `use_api=True` in your code.

## 📈 Scaling Path

### Week 1: Manual Processing
- Use CLI to process CSVs
- Generate reports on demand
- Test with sample data

### Week 2-4: Pilot with Real Data
- Export ET comments weekly
- Process via CLI
- Generate weekly reports

### Month 2: Semi-Automation
- Set up Apify scrapers
- Auto-download CSVs
- Batch processing

### Month 3+: Full Automation
- API integration
- Scheduled processing
- Email delivery
- Web dashboard

## 🔧 Troubleshooting

**System won't run?**
→ Check requirements: `pip install -r requirements.txt`

**No entities detected?**
→ Check `known_entities.json` and `config.py` seed list

**Report looks wrong?**
→ Edit `report_generator.py` to customize

**API too expensive?**
→ Use rule-based mode (set `use_api=False`)

## 💡 Pro Tips

1. **Start with sample data** - Test before using real data
2. **Process weekly** - More frequent = better trends
3. **Review entity database** - Add missed celebrities manually
4. **Adjust thresholds** - Fine-tune alerts based on your needs
5. **Keep post_subject filled** - Helps entity detection

## 📞 Need Help?

1. Read **QUICKSTART.md** - Solves 80% of issues
2. Check **README.md** - Full documentation
3. Review **PROJECT_SUMMARY.md** - Understand the system
4. Run **test_system.py** - Verify installation

## ✅ Success Checklist

Before going live, ensure:
- [ ] `test_system.py` runs successfully
- [ ] Sample PDF report looks good
- [ ] CSV format matches your data
- [ ] Entity detection working
- [ ] Thresholds make sense
- [ ] Report format approved

## 🎪 What's Included

- ✅ Complete working system
- ✅ Interactive CLI
- ✅ Professional PDF reports
- ✅ Sample data for testing
- ✅ Full documentation
- ✅ Configuration examples
- ✅ Test scripts
- ✅ Entity database
- ✅ Scalable architecture

## 🚀 Get Started

Run this now:
```bash
python3 test_system.py
```

Then read **LAUNCH_PLAN.md** for next steps.

---

**Built by Joe | CBS Marketing**
**November 2024**

*Transform social engagement into strategic intelligence*
