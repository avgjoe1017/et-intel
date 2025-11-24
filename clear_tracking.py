#!/usr/bin/env python3
"""Clear processed tracking for specific CSV files"""
import json
from pathlib import Path

tracker_path = Path('et_intel/data/database/processed_csvs.json')
if tracker_path.exists():
    data = json.load(open(tracker_path))
    files_to_clear = [
        'dataset_instagram-comments-scraper_2025-11-23_05-10-02-585.csv',
        'dataset_instagram-comments-scraper_2025-11-23_05-16-50-355.csv'
    ]
    
    # Remove entries that match our files
    new_data = {}
    for k, v in data.items():
        filename = v.get('filename', '')
        path = v.get('path', '')
        if not any(f in filename or f in path for f in files_to_clear):
            new_data[k] = v
    
    json.dump(new_data, open(tracker_path, 'w'), indent=2)
    print(f"Cleared {len(data) - len(new_data)} entries. {len(new_data)} entries remaining.")
else:
    print("No tracking file found.")

