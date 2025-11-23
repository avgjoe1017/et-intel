# URL & CAPTION STORAGE GUIDE

## Quick Answer

**Yes, we store the URL (and caption) from ESUIT files!**

---

## What Gets Stored

### From ESUIT Format

**Line 1**: Post URL  
**Lines 2-N**: Caption text  
**After header**: Comments

All three are extracted and stored.

---

## Where It's Stored

### 1. In the Cleaned CSV (Primary)

When you run the preprocessor:
```bash
python preprocess_esuit.py YOUR_FILE.csv
```

**Output CSV includes**:
- `post_url` column (same for all rows)
- `post_caption` column (same for all rows)
- All comment data

**Example row**:
```csv
username,comment,timestamp,likes,post_url,post_caption
user123,"Love this! 🔥",2024-11-20,145,https://instagram.com/p/ABC,Full caption text here...
```

---

### 2. In the Metadata File (Reference)

Also creates a separate text file:
```
Post URL: https://www.instagram.com/p/DRSz8XLgVCC/?img_index=1
Caption: A 'Gossip Girl' and 'Gilmore' guy in a Christmas rom-com?...
Total Comments: 382
```

**Location**: `data/uploads/FILENAME_metadata.txt`

---

### 3. In the Processed Data (Final)

After you upload to the system:
```bash
python cli.py --import cleaned.csv --platform instagram
```

**The processed CSV includes**:
- `post_url` - Instagram URL
- `post_caption` - Full caption text
- `post_subject` - Your provided subject
- All sentiment analysis
- All entity extraction

**Location**: `data/processed/processed_instagram_*.csv`

---

## Why This Matters

### For Analysis

**Track posts over time:**
```python
import pandas as pd

# Load all processed data
df = pd.read_csv('data/processed/processed_instagram_*.csv')

# Group by post
by_post = df.groupby('post_url').agg({
    'sentiment_score': 'mean',
    'comment_text': 'count'
})

print("Post performance:")
print(by_post)
```

**Compare posts:**
- Which posts get most engagement?
- Which posts get most positive sentiment?
- Which celebrities drive most comments?

---

### For Reporting

**Intelligence briefs can show:**
- "Post X got 382 comments with +0.612 sentiment"
- "Here's the URL to view: [link]"
- "Caption: [full text]"

**For stakeholders:**
- Click through to see actual post
- Verify caption matches analysis
- Check if comments match post topic

---

### For Auditing

**Traceable data:**
- Every comment links back to source post
- Can verify sentiment with original context
- Reproducible analysis

**Example**:
```
Comment: "She ate 🔥"
Sentiment: Excitement (+1.00)
From post: https://instagram.com/p/DRSz8XLgVCC/
Caption: "Leighton Meester and Jared Padalecki..."
```

You can click the URL and verify the comment is real.

---

## How It Flows Through the System

### Step 1: ESUIT Export
```
ESUIT file:
  Line 1: https://instagram.com/p/ABC
  Lines 2-5: Caption text here...
  Line 6: "Id","UserId","Author"...
  Lines 7+: Comments
```

### Step 2: Preprocessing
```bash
python preprocess_esuit.py FILE.csv
```
**Creates**:
- `cleaned.csv` with `post_url` and `post_caption` columns
- `metadata.txt` with URL and caption

### Step 3: Upload
```bash
python cli.py --import cleaned.csv --platform instagram
```
**Preserves**:
- `post_url` → carried through to processed data
- `post_caption` → carried through to processed data

### Step 4: Analysis
**Available in**:
- Processed CSV: `data/processed/*.csv`
- Intelligence brief JSON
- Reports can reference URLs

---

## Real Example (Your File)

### Original ESUIT File
```
https://www.instagram.com/p/DRSz8XLgVCC/?img_index=1
A 'Gossip Girl' and 'Gilmore' guy in a Christmas rom-com?...
[5 lines of caption]
"Id","UserId","Author","Content"...
[382 comments]
```

### After Preprocessing
**File**: `leighton_meester_cleaned.csv`
- Column `post_url`: `https://www.instagram.com/p/DRSz8XLgVCC/?img_index=1`
- Column `post_caption`: Full 5-line caption
- 382 rows (one per comment)

### After Upload
**File**: `processed_instagram_*.csv`
- All 382 comments
- Each has `post_url` and `post_caption`
- Plus sentiment, entities, etc.

### In Intelligence Brief
```json
{
  "metadata": {
    "post_urls": [
      "https://www.instagram.com/p/DRSz8XLgVCC/?img_index=1"
    ]
  },
  "posts": [
    {
      "url": "https://www.instagram.com/p/DRSz8XLgVCC/",
      "caption": "A 'Gossip Girl' and 'Gilmore' guy...",
      "total_comments": 382,
      "avg_sentiment": 0.612
    }
  ]
}
```

---

## Multiple Posts

### Batch Processing

If you process multiple ESUIT files:
```bash
python preprocess_esuit.py post1.csv
python preprocess_esuit.py post2.csv
python preprocess_esuit.py post3.csv

python cli.py --batch data/uploads/*_cleaned.csv --platform instagram
```

**Each comment tracks its source post:**
```
Comment 1 → post_url: https://instagram.com/p/ABC
Comment 2 → post_url: https://instagram.com/p/ABC
Comment 3 → post_url: https://instagram.com/p/DEF
Comment 4 → post_url: https://instagram.com/p/DEF
```

**You can then analyze**:
- Sentiment by post
- Engagement by post
- Entities by post
- Compare posts

---

## Query Examples

### Find all comments from a specific post
```python
import pandas as pd

df = pd.read_csv('data/processed/processed_instagram_*.csv')

post_url = "https://www.instagram.com/p/DRSz8XLgVCC/"
post_comments = df[df['post_url'] == post_url]

print(f"Comments: {len(post_comments)}")
print(f"Avg sentiment: {post_comments['sentiment_score'].mean()}")
```

### Compare two posts
```python
post1 = df[df['post_url'] == "https://instagram.com/p/POST1"]
post2 = df[df['post_url'] == "https://instagram.com/p/POST2"]

print(f"Post 1: {len(post1)} comments, {post1['sentiment_score'].mean():.2f} sentiment")
print(f"Post 2: {len(post2)} comments, {post2['sentiment_score'].mean():.2f} sentiment")
```

### Find posts with high engagement
```python
by_post = df.groupby('post_url').agg({
    'comment_text': 'count',
    'sentiment_score': 'mean'
}).sort_values('comment_text', ascending=False)

print("Top posts by comments:")
print(by_post.head())
```

---

## Caption Usage

### Full Caption is Available

**For context in analysis:**
- What was the post actually about?
- Does sentiment match the caption tone?
- Are comments responding to specific caption points?

**Example**:
```python
df = pd.read_csv('processed.csv')

# Show caption with sentiment
for url in df['post_url'].unique():
    post = df[df['post_url'] == url]
    caption = post['post_caption'].iloc[0]
    avg_sent = post['sentiment_score'].mean()
    
    print(f"Caption: {caption[:100]}...")
    print(f"Sentiment: {avg_sent:.2f}")
    print()
```

---

## Benefits

### ✅ Traceability
Every comment links back to source post

### ✅ Verification
Stakeholders can click URLs to verify

### ✅ Comparison
Analyze post performance side-by-side

### ✅ Context
Caption provides context for sentiment

### ✅ Reporting
Include URLs in intelligence briefs

---

## Summary

**Yes, URLs (and captions) are stored everywhere:**
- ✅ Cleaned CSV from preprocessor
- ✅ Metadata text file
- ✅ Processed data after upload
- ✅ Available for analysis and reporting

**Every comment is traceable to its source post.**

---

## Quick Reference

### Check URL in cleaned file
```bash
python3 -c "import pandas as pd; df = pd.read_csv('cleaned.csv'); print(df['post_url'].iloc[0])"
```

### Check URL in processed file
```bash
python3 -c "import pandas as pd; df = pd.read_csv('data/processed/processed_instagram_*.csv'); print(df['post_url'].value_counts())"
```

### View metadata
```bash
cat data/uploads/FILENAME_metadata.txt
```

---

**URLs are tracked at every step.** 🎯
