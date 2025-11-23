# ET SOCIAL INTELLIGENCE - QUICK START GUIDE

## 🚀 Get Started in 5 Minutes

### Step 1: Verify Installation
```bash
cd /home/claude/et_social_intelligence
python3 test_system.py
```

If you see "✓ System is operational!" you're ready to go.

### Step 2: Run the Interactive Interface
```bash
python3 cli.py
```

### Step 3: Import Your First CSV
1. Select **Option 1** (Import CSV)
2. Enter path to your CSV file
3. Choose platform (Instagram or YouTube)
4. Add post metadata (URL, subject)
5. Wait for processing to complete

### Step 4: Generate Your First Report
1. Select **Option 2** (Generate Brief)
2. Choose date range and filters
3. Wait for PDF generation
4. Find your report in `reports/` folder

---

## 📋 CSV Format Checklist

Your CSV needs these columns (names are flexible):
- ✅ Username/Author
- ✅ Comment/Text  
- ✅ Timestamp/Date
- ⚪ Likes (optional)

**Example:**
```csv
username,comment,timestamp,likes
@user123,This is amazing! 🔥,2024-11-20 14:30:00,245
@moviefan,I'm obsessed with this,2024-11-20 14:35:00,89
```

---

## 🎯 First Time Workflow

### Monday: Collect Data
Export comments from your top ET posts:
- **Instagram**: Use Apify or export tool
- **YouTube**: Use comment scraper or API

Save as CSV files.

### Tuesday: Process Data
```bash
python3 cli.py
# Import all CSV files one by one
```

### Wednesday: Generate Report
```bash
python3 cli.py
# Generate intelligence brief
# Apply filters (last 30 days, Instagram only, etc.)
```

### Thursday: Review & Share
- Open PDF in `reports/` folder
- Review velocity alerts
- Add executive notes
- Share with stakeholders

---

## 💡 Pro Tips

### Get Better Results
1. **Always provide post_subject** - helps entity detection
2. **Process regularly** - weekly is ideal for trend tracking
3. **Review entity database** - in `data/database/known_entities.json`
4. **Add seed relationships** - in `config.py` for couples/storylines

### Cost Management (API Mode)
- Start with **rule-based mode** (free) for testing
- Enable **API mode** for production reports
- Batch process weekly instead of daily
- Expected cost: **$5-15/month** for 10K comments

### Entity Detection Issues?
If system isn't catching entities:
1. Lower `MIN_ENTITY_MENTIONS` in `config.py`
2. Manually add entities to `known_entities.json`
3. Provide detailed `post_subject` when importing

---

## 🔧 Common Issues

**"Could not detect required columns"**
→ Check CSV has username, comment, timestamp columns

**"No processed data found"**  
→ Import CSV files first (Option 1)

**Entities not detected**
→ Check post_subject is filled in during import

**API errors**
→ Verify OpenAI key: `echo $OPENAI_API_KEY`

---

## 📊 What You Get

### PDF Intelligence Brief Includes:
1. **Executive Summary** - Key findings at a glance
2. **Velocity Alerts** - Entities with sentiment spikes/drops
3. **Top Entities** - Ranked by volume and sentiment
4. **Emotion Distribution** - What people are feeling
5. **Active Storylines** - Trending topics (lawsuits, relationships, etc.)
6. **Demographics** - Engagement patterns and timing

### JSON Data File Includes:
- Raw entity data
- Sentiment metrics
- Velocity calculations
- Demographic breakdowns
- Can be imported into other tools

---

## 📈 Scaling Up

### From MVP to Production

**Week 1-2: Manual Processing**
- Export CSVs manually/via VA
- Process weekly via CLI
- Validate outputs

**Week 3-4: Semi-Automation**  
- Set up Apify scheduled scrapers
- Auto-download CSVs to folder
- Process in batches

**Month 2: Full Automation**
- Integrate Instagram/YouTube APIs
- Schedule automated processing
- Email report delivery

---

## 🎓 Understanding the Outputs

### Sentiment Scores
- **+0.5 to +1.0**: Very Positive
- **+0.2 to +0.5**: Positive
- **-0.2 to +0.2**: Neutral
- **-0.5 to -0.2**: Negative
- **-1.0 to -0.5**: Very Negative

### Velocity Alerts
- **30%+ change in 72 hours** = Alert triggered
- **Rising** ↑ = Sentiment improving
- **Falling** ⚠ = Sentiment declining

### Emotions Tracked
- **Excitement**: "I can't even", "she ate", "iconic"
- **Love**: "adorable", "ship them", "goals"
- **Anger**: "furious", "terrible", "wtf"
- **Disappointment**: "expected better", "let down"
- **Fatigue**: "move on", "over it", "tired of"
- **Disgust**: "gross", "cringe", "ew"
- **Surprise**: "wait what", "didn't see coming"
- **Neutral**: Informational, no strong emotion

---

## 🎪 Sample Use Cases

### Use Case 1: Talent Risk Assessment
**Goal**: Decide if Blake Lively is safe for new CBS project

**Process**:
1. Import last 90 days of ET comments mentioning Blake
2. Generate brief with focus on Blake Lively
3. Check velocity alerts and emotion breakdown
4. Compare to other talent options

**Decision**: If sentiment stable and positive → green light

---

### Use Case 2: Marketing Message Testing
**Goal**: See if "Awards Season Starts Now" resonates

**Process**:
1. Track mentions of awards/Oscars/Golden Globes
2. Monitor sentiment around awards content vs other content
3. Check for fatigue signals
4. Compare engagement on awards posts

**Decision**: High engagement + low fatigue → double down

---

### Use Case 3: Controversy Monitoring
**Goal**: Assess if "It Ends With Us" lawsuit is damaging

**Process**:
1. Set velocity alerts for Blake + Justin Baldoni
2. Track "lawsuit" storyline mentions
3. Monitor sentiment changes daily
4. Compare to pre-lawsuit baseline

**Decision**: If velocity drops >30% → damage control needed

---

## 📞 Need Help?

1. **Check README.md** - Full documentation
2. **Review test_system.py** - Example code
3. **Inspect sample_data.csv** - Reference format
4. **Read config.py** - All settings explained

---

**Ready to transform social comments into strategic intelligence?**

Run `python3 cli.py` and let's go! 🚀
