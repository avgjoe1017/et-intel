# Quick Start Guide - Enhanced Features

## 🚀 New Features Added

### 1. Hugging Face Emotion Classifier (FREE!)
- **90%+ accuracy** emotion detection
- **No API costs** - runs locally
- **Automatic** - enabled by default

### 2. Streamlit Dashboard
- **Interactive web app** for exploring data
- **Real-time filtering** and charts
- **Self-service analytics** for stakeholders

### 3. spaCy Entity Extraction
- **Better entity detection** (catches nicknames, variations)
- **Proper NER** instead of regex patterns

### 4. NetworkX Relationship Graphs
- **Visual relationship maps** between entities
- **Centrality metrics** to find influential entities

### 5. SQLite Database
- **Faster queries** on large datasets
- **Better scalability** for millions of comments

---

## 📦 Installation

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install spaCy Model
```bash
python -m spacy download en_core_web_lg
```

**Note**: This downloads ~500MB. For smaller install, use:
```bash
python -m spacy download en_core_web_sm  # ~40MB
```

### Step 3: First Run (Downloads Models)
The first time you use Hugging Face, it will download the emotion model (~500MB). This happens automatically.

---

## 🎯 Quick Start

### Option 1: Process Data with Enhanced Features
```bash
# Process CSVs (uses HF emotion classifier automatically)
python -m et_intel.cli.cli --batch
```

### Option 2: Launch Interactive Dashboard
```bash
# First, process some data
python -m et_intel.cli.cli --batch

# Then launch dashboard
streamlit run streamlit_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📊 Using the Dashboard

1. **Filters** (left sidebar):
   - Select platforms (Instagram, YouTube)
   - Choose date range
   
2. **Tabs**:
   - **Sentiment Trends**: See sentiment over time
   - **Entities**: View top entities and sample comments
   - **Emotion Breakdown**: Emotion distribution and trends
   - **Relationships**: Entity relationship networks

3. **Metrics** (top):
   - Total comments
   - Average sentiment
   - Number of platforms
   - Unique posts

---

## 🔧 Advanced Usage

### Use Hugging Face Emotion Classifier
```python
from et_intel import ETIntelligencePipeline

# Automatically uses HF (no API needed)
pipeline = ETIntelligencePipeline(use_api=False)

# Process data
df, entities = pipeline.process_new_data(
    csv_path='comments.csv',
    platform='instagram',
    post_metadata={'subject': 'Taylor Swift'}
)
```

### Create Relationship Graphs
```python
from et_intel import RelationshipGraph, EntityExtractor
import pandas as pd

# Load processed data
from et_intel.core.ingestion import CommentIngester
ingester = CommentIngester()
df = ingester.load_all_processed()

# Extract entities
extractor = EntityExtractor()
entities = extractor.extract_entities_from_comments(df)

# Build graph
graph_builder = RelationshipGraph()
graph = graph_builder.build_graph_from_entities(entities, df)

# Visualize
graph_builder.visualize()

# Get centrality metrics
metrics = graph_builder.get_centrality_metrics()
print("Most influential entities:", metrics['degree'])
```

### Query SQLite Database
```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///et_intel/data/database/et_intelligence.db')

# Query
df = pd.read_sql("""
    SELECT entity, AVG(sentiment_score) as avg_sentiment
    FROM comments
    WHERE timestamp > '2024-01-01'
    GROUP BY entity
    ORDER BY avg_sentiment DESC
    LIMIT 10
""", engine)
```

---

## 💡 Tips

### Performance
- **First run**: Hugging Face downloads models (~500MB)
- **Subsequent runs**: Much faster (models cached)
- **Database**: Faster than CSV for large datasets

### Accuracy
- **Hugging Face**: Best emotion accuracy (90%+)
- **spaCy**: Best entity detection
- **TextBlob**: Good baseline for comparison
- **Ensemble**: Combines methods for confidence

### Cost
- **All new features**: FREE (run locally)
- **No API costs**: Hugging Face and spaCy are free
- **Optional OpenAI**: Still available for even better accuracy

---

## 🐛 Troubleshooting

### "spaCy model not found"
```bash
python -m spacy download en_core_web_lg
```

### "Hugging Face model download slow"
- First download takes time (~500MB)
- Models are cached after first use
- Use `--no-hf` flag to disable if needed

### "Streamlit dashboard shows no data"
- Make sure you've processed data first:
  ```bash
  python -m et_intel.cli.cli --batch
  ```

### "Database errors"
- Database is optional - falls back to CSV
- Check write permissions in `data/database/` folder

---

## 📈 What's Next?

### Recommended Workflow
1. **Week 1**: Process data with enhanced features
2. **Week 2**: Explore data in Streamlit dashboard
3. **Week 3**: Create relationship graphs for key entities
4. **Week 4**: Set up scheduled processing

### Future Enhancements
- Email scheduled reports
- Automated weekly processing
- Multi-language support (spaCy supports many languages)
- Custom emotion models fine-tuned on entertainment data

---

**Questions?** Check the main README.md or open an issue.

