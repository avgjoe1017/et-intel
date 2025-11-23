"""
ET Social Intelligence System - Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in project root (parent of et_intel package)
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    # Also try loading from current directory (if running from project root)
    load_dotenv()

# System version
SYSTEM_VERSION = "1.0.0"
CONFIG_VERSION = "1.0"

# Python version requirement
REQUIRED_PYTHON = "3.8+"  # Minimum Python version

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"
DB_DIR = DATA_DIR / "database"

# Create directories if they don't exist
for dir_path in [DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, REPORTS_DIR, DB_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# API Configuration (loaded from .env file or environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # For sentiment analysis

# Model settings (using cheaper models to stay under $100/month)
SENTIMENT_MODEL = "gpt-4o-mini"  # Cheaper than GPT-4, good enough for sentiment
ENTITY_MODEL = "gpt-4o-mini"
MAX_TOKENS_SENTIMENT = 150
MAX_TOKENS_ENTITY = 300

# Processing settings
BATCH_SIZE = 50  # Process comments in batches to manage API costs
MIN_COMMENT_LENGTH = 1  # Keep even single emoji/character comments (emojis ARE sentiment!)
MAX_COMMENTS_PER_POST = 500  # Cap to manage costs
MAX_COMMENTS_PER_BRIEF = 10000  # Maximum comments to include in a single brief

# Minimum sample sizes for statistical validity
MIN_VELOCITY_SAMPLE_SIZE = 10  # Minimum comments needed to calculate velocity
MIN_ENTITY_SAMPLE_SIZE = 5  # Minimum mentions to consider entity significant

# Entity detection settings
MIN_ENTITY_MENTIONS = 3  # Minimum mentions to consider an entity "trending"
COUPLE_THRESHOLD = 0.6  # If two entities co-occur in 60%+ of mentions, flag as couple

# Sentiment categories
EMOTIONS = [
    "excitement",
    "anger", 
    "disappointment",
    "love",
    "disgust",
    "surprise",
    "fatigue",  # For tracking "storyline fatigue"
    "neutral"
]

# Velocity alert thresholds
VELOCITY_WINDOW_HOURS = 72  # Track sentiment changes over 3 days
VELOCITY_ALERT_THRESHOLD = 0.3  # Alert if sentiment drops/rises 30%+ 

# Seed list for couples/storylines (manually maintained)
SEED_RELATIONSHIPS = [
    ["Travis Kelce", "Taylor Swift"],
    ["Blake Lively", "It Ends With Us"],
    ["Justin Baldoni", "It Ends With Us"],
    # Add more as needed
]

# Report settings
REPORT_TITLE = "ET Social Intelligence Brief"
REPORT_SUBTITLE = "Mass Market Sentiment Analysis"

# Logging configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "et_intelligence.log"  # Set to None to disable file logging


