#!/usr/bin/env python3
"""
Apify Instagram Scraper Format Preprocessor
Handles the specific format from Apify Instagram Comments Scraper
Auto-detects header location (variable caption length)
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def preprocess_apify_instagram(input_path: str, output_path: str = None) -> str:
    """
    Preprocess Apify Instagram scraper format
    
    Format:
    - Lines 1-N: URL and caption (VARIABLE LENGTH - could be 2 lines, could be 20 lines)
    - After that: Header row + comments
    
    Args:
        input_path: Path to Apify CSV
        output_path: Where to save cleaned CSV (default: auto-generated)
    
    Returns:
        Path to cleaned CSV
    """
    input_file = Path(input_path)
    
    if output_path is None:
        output_path = input_file.parent / f"{input_file.stem}_cleaned.csv"
    
    print(f"Processing Apify Instagram format CSV: {input_path}")
    
    # Read entire file to find header row
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extract URL from line 1 (remove BOM if present)
    post_url = lines[0].strip().replace('﻿', '').replace('\ufeff', '')
    print(f"Post URL: {post_url}")
    
    # Find header row (contains comment_like_count and text columns)
    header_index = None
    for i, line in enumerate(lines):
        if 'comment_like_count' in line and '"text"' in line:
            header_index = i
            break
    
    if header_index is None:
        raise ValueError("Could not find Apify header row in CSV (looking for 'comment_like_count' and 'text' columns)")
    
    # Caption is everything between line 1 and header
    caption_lines = [line.strip() for line in lines[1:header_index] if line.strip()]
    caption = ' '.join(caption_lines)
    print(f"Caption: {caption[:100]}...")
    print(f"Caption lines: {len(caption_lines)}")
    print(f"Header was on line: {header_index + 1}")
    
    # Read comments starting from header row
    # We need to handle multi-line fields and encoding properly
    try:
        # First, read just to get the header row to understand column structure
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            header_line = lines[header_index]
        
        # Now read CSV starting from header row with proper encoding
        df = pd.read_csv(
            input_path,
            skiprows=header_index,
            encoding='utf-8',
            on_bad_lines='skip',
            engine='python',
            quotechar='"',
            escapechar='\\'
        )
    except Exception as e:
        # Try with C engine and different parameters
        try:
            df = pd.read_csv(
                input_path,
                skiprows=header_index,
                encoding='utf-8',
                on_bad_lines='skip',
                engine='c',
                quotechar='"'
            )
        except Exception as e2:
            raise ValueError(f"Could not read CSV starting from header row: {e}. Also tried: {e2}")
    
    print(f"Found {len(df)} comments")
    
    # Standardize column names
    # Apify format uses: user/username, text, created_at, comment_like_count
    cleaned = pd.DataFrame({
        'username': df['user/username'] if 'user/username' in df.columns else df.get('username', ''),
        'comment': df['text'],
        'timestamp': df['created_at'],
        'likes': df['comment_like_count'],
        'post_url': post_url,  # Add post URL to every row for reference
        'post_caption': caption  # Add caption to every row for context
    })
    
    # Save cleaned CSV with UTF-8 encoding
    cleaned.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"[OK] Processed {len(cleaned)} comments")
    print(f"  URL: {post_url}")
    print(f"  Caption lines: {len(caption_lines)}")
    print(f"  Header was on line: {header_index + 1}")
    print(f"[OK] Cleaned CSV saved: {output_path}")
    print(f"  Columns: {list(cleaned.columns)}")
    
    return str(output_path)


def detect_apify_format(csv_path: str) -> bool:
    """
    Check if CSV is in Apify Instagram scraper format
    
    Returns:
        True if Apify format detected
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Check if line 1 is Instagram URL
        if len(lines) == 0:
            return False
        
        first_line = lines[0].strip().replace('﻿', '').replace('\ufeff', '')
        if 'instagram.com/p/' not in first_line and 'instagram.com/reel/' not in first_line:
            return False
        
        # Look for Apify columns in first 50 lines
        for line in lines[:50]:
            if 'comment_like_count' in line and '"text"' in line:
                return True
        
        return False
    except Exception as e:
        logger.error(f"Error detecting Apify format: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python preprocess_apify.py <input_csv> [output_csv]")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not detect_apify_format(input_csv):
        print("⚠ Warning: This doesn't appear to be Apify format")
        proceed = input("Continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            sys.exit(1)
    
    cleaned_path = preprocess_apify_instagram(input_csv, output_csv)
    
    print(f"\n✅ SUCCESS!")
    print(f"\nNow you can process with:")
    print(f"python -m et_intel.cli.cli --batch")

