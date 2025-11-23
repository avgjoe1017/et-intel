#!/usr/bin/env python3
"""
Instagram ESUIT Format Preprocessor
Handles the specific format from ESUIT Comments Exporter
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def preprocess_esuit_csv(input_path: str, output_path: str = None) -> str:
    """
    Preprocess ESUIT Comments Exporter CSV format
    
    Format:
    - Line 1: Post URL
    - Lines 2-N: Caption text
    - Line with "Id","UserId"...: Header row
    - Remaining lines: Comments
    
    Args:
        input_path: Path to ESUIT CSV
        output_path: Where to save cleaned CSV (default: auto-generated)
    
    Returns:
        Path to cleaned CSV
    """
    input_file = Path(input_path)
    
    if output_path is None:
        output_path = input_file.parent / f"{input_file.stem}_cleaned.csv"
    
    print(f"Processing ESUIT format CSV: {input_path}")
    
    # Read the entire file
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extract metadata
    post_url = lines[0].strip()
    print(f"Post URL: {post_url}")
    
    # Find the header row (contains "Id","UserId","Author","Content"...)
    header_index = None
    for i, line in enumerate(lines):
        if '"Id"' in line and '"Author"' in line and '"Content"' in line:
            header_index = i
            break
    
    if header_index is None:
        raise ValueError("Could not find comment header row in CSV")
    
    # Caption is everything between line 1 and header
    caption_lines = [line.strip() for line in lines[1:header_index] if line.strip()]
    caption = ' '.join(caption_lines)
    print(f"Caption: {caption[:100]}...")
    
    # Read comments starting from header row
    df = pd.read_csv(input_path, skiprows=header_index)
    
    print(f"Found {len(df)} comments")
    
    # Standardize column names
    cleaned = pd.DataFrame({
        'username': df['Author'],
        'comment': df['Content'],
        'timestamp': df['CommentAt'],
        'likes': df['ReactionsCount'],
        'comment_id': df['Id'],
        'user_id': df['UserId'],
        'avatar': df.get('Avatar', ''),
        'depth': df.get('Depth', 0),
        'sub_comments': df.get('SubCommentsCount', 0),
        'post_url': post_url,  # Add post URL to every row for reference
        'post_caption': caption  # Add caption to every row for context
    })
    
    # Save cleaned CSV
    cleaned.to_csv(output_path, index=False)
    
    print(f"✓ Cleaned CSV saved: {output_path}")
    print(f"  Total comments: {len(cleaned)}")
    print(f"  Columns: {list(cleaned.columns)}")
    
    # Also save metadata as separate file
    metadata_path = Path(output_path).parent / f"{Path(output_path).stem}_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"Post URL: {post_url}\n")
        f.write(f"Caption: {caption}\n")
        f.write(f"Total Comments: {len(cleaned)}\n")
    
    print(f"✓ Metadata saved: {metadata_path}")
    
    return str(output_path)


def detect_esuit_format(csv_path: str) -> bool:
    """
    Check if CSV is in ESUIT format
    
    Returns:
        True if ESUIT format detected
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Check if line 1 is an Instagram URL
        if len(lines) > 0 and 'instagram.com' in lines[0]:
            # Check if we can find the ESUIT header
            for line in lines:
                if '"Id"' in line and '"Author"' in line and '"Content"' in line:
                    return True
        
        return False
    except Exception as e:
        logger.error(f"Error detecting ESUIT format: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python preprocess_esuit.py <input_csv> [output_csv]")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not detect_esuit_format(input_csv):
        print("⚠ Warning: This doesn't appear to be ESUIT format")
        proceed = input("Continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            sys.exit(1)
    
    cleaned_path = preprocess_esuit_csv(input_csv, output_csv)
    
    print(f"\n✅ SUCCESS!")
    print(f"\nNow you can upload with:")
    print(f"python cli.py --import {cleaned_path} --platform instagram --subject 'Leighton Meester'")
