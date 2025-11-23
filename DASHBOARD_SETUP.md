# Streamlit Dashboard Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Process Some Data
Before running the dashboard, you need processed data:
```bash
# Process CSVs in uploads folder
python -m et_intel.cli.cli --batch
```

### 3. Launch Dashboard
```bash
streamlit run streamlit_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📊 Dashboard Features

### Main Metrics
- **Total Comments**: Number of processed comments
- **Avg Sentiment**: Average sentiment score (-1 to +1)
- **Platforms**: Number of platforms (Instagram, YouTube)
- **Unique Posts**: Number of unique posts analyzed

### Tabs

#### 1. Sentiment Trends
- **Daily Average Sentiment**: Line chart showing sentiment over time
- **Sentiment Distribution**: Histogram of sentiment scores
- **Emotion Distribution**: Pie chart of primary emotions

#### 2. Entities
- **Entity Mentions**: Top entities mentioned in comments
- **Sample Comments**: Browse actual comments with sentiment scores

#### 3. Emotion Breakdown
- **Emotion Timeline**: Trends of different emotions over time
- **Emotion by Platform**: Heatmap showing emotion distribution by platform

#### 4. Relationships
- **Entity Relationship Networks**: Visual graphs of entity connections
- *Note: Requires entity extraction during processing*

### Filters (Sidebar)
- **Platform**: Filter by Instagram, YouTube, or both
- **Date Range**: Select specific date range for analysis

---

## 🔧 Configuration

### Dashboard Settings
Edit `streamlit_dashboard.py` to customize:
- Cache duration (default: 5 minutes)
- Chart colors and styles
- Default date ranges
- Number of items displayed

### Data Source
The dashboard reads from:
- **Primary**: SQLite database (`data/database/et_intelligence.db`)
- **Fallback**: CSV files in `data/processed/`

---

## 🐛 Troubleshooting

### "No processed data found"
**Solution**: Process data first:
```bash
python -m et_intel.cli.cli --batch
```

### Dashboard shows old data
**Solution**: Clear Streamlit cache:
```bash
# Stop dashboard (Ctrl+C)
# Delete cache
rm -rf ~/.streamlit/cache  # Linux/Mac
# Or manually delete cache folder
```

### Port already in use
**Solution**: Use different port:
```bash
streamlit run streamlit_dashboard.py --server.port 8502
```

### Charts not displaying
**Solution**: Check that processed data has required columns:
- `timestamp`
- `sentiment_score`
- `primary_emotion`
- `platform`

---

## 📈 Performance Tips

### Large Datasets
For datasets with 100K+ comments:
1. Use SQLite database (automatic)
2. Increase cache duration in dashboard
3. Add date filters to reduce data loaded

### Slow Loading
1. Check database is being used (not CSV fallback)
2. Reduce date range in filters
3. Process data in smaller batches

---

## 🚢 Deployment

### Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repository
4. Deploy!

### Local Network
Share with team:
```bash
streamlit run streamlit_dashboard.py --server.address 0.0.0.0
```
Then access from other machines: `http://YOUR_IP:8501`

### Docker (Optional)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "streamlit_dashboard.py", "--server.address", "0.0.0.0"]
```

---

## 🎨 Customization

### Add New Charts
Edit `streamlit_dashboard.py` and add to tabs:
```python
with tab1:
    # Your new chart code
    fig = px.bar(...)
    st.plotly_chart(fig)
```

### Change Colors
Update Plotly theme:
```python
import plotly.io as pio
pio.templates.default = "plotly_dark"  # or "plotly", "ggplot2", etc.
```

### Add Filters
Add to sidebar:
```python
filter_value = st.sidebar.selectbox("Filter", options)
df_filtered = df[df['column'] == filter_value]
```

---

## 📝 Notes

- Dashboard auto-refreshes when data changes (cache expires after 5 min)
- All charts are interactive (zoom, pan, hover)
- Data is cached for performance
- Dashboard works with both CSV and database storage

---

## 🔗 Related Files

- Main dashboard: `streamlit_dashboard.py`
- Data loading: `et_intel/core/ingestion.py`
- Tests: `tests/test_dashboard.py`

---

**Questions?** Check the main README.md or open an issue.

