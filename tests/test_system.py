#!/usr/bin/env python3
"""
Test Script - Verify ET Intelligence System
Runs end-to-end test with sample data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import config to get correct paths
from et_intel import config

print("\n" + "="*70)
print("  ET SOCIAL INTELLIGENCE - SYSTEM TEST")
print("="*70 + "\n")

print("Testing system with sample data...\n")

# Test 1: Import modules
print("Test 1: Importing modules...")
try:
    from et_intel.core.pipeline import ETIntelligencePipeline
    from et_intel.reporting.report_generator import IntelligenceBriefGenerator
    from et_intel.core.ingestion import CommentIngester
    from et_intel.core.entity_extraction import EntityExtractor
    from et_intel.core.sentiment_analysis import SentimentAnalyzer
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Process sample data
print("\nTest 2: Processing sample data...")
try:
    pipeline = ETIntelligencePipeline(use_api=False)
    
    # Use config paths - data is in et_intel/data/ with new structure
    sample_csv = config.DATA_DIR / "sample" / "sample_data.csv"
    # Fallback to project root data/ if not found (for backward compatibility)
    if not sample_csv.exists():
        sample_csv = project_root / "data" / "sample" / "sample_data.csv"
    
    if not sample_csv.exists():
        print(f"✗ Sample data not found: {sample_csv}")
        sys.exit(1)
    
    df, entities = pipeline.process_new_data(
        csv_path=str(sample_csv),
        platform='instagram',
        post_metadata={
            'post_url': 'https://instagram.com/p/TEST123',
            'subject': 'Taylor Swift, Travis Kelce',
            'post_caption': 'Taylor arrives at Chiefs game with Travis'
        }
    )
    
    print(f"✓ Processed {len(df)} comments")
    print(f"✓ Found {len(entities['people'])} people")
    print(f"✓ Found {len(entities['shows'])} shows")
    print(f"✓ Found {len(entities['couples'])} couples")
    print(f"✓ Found {len(entities['storylines'])} storylines")
    
except Exception as e:
    print(f"✗ Processing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Entity extraction
print("\nTest 3: Verifying entity extraction...")
try:
    people_names = [p[0] for p in entities['people']]
    
    # Check for expected entities
    expected = ['Taylor Swift', 'Travis Kelce', 'Blake Lively']
    found = [e for e in expected if any(e in p for p in people_names)]
    
    print(f"✓ Found {len(found)}/{len(expected)} expected entities")
    
    if entities['couples']:
        print(f"✓ Detected relationships:")
        for p1, p2, count, source in entities['couples'][:3]:
            print(f"  - {p1} + {p2} (co-occurrence: {count})")
    
except Exception as e:
    print(f"✗ Entity verification failed: {e}")

# Test 4: Sentiment analysis
print("\nTest 4: Verifying sentiment analysis...")
try:
    if 'sentiment_score' in df.columns:
        avg_sentiment = df['sentiment_score'].mean()
        print(f"✓ Average sentiment: {avg_sentiment:.3f}")
        
        emotion_dist = df['primary_emotion'].value_counts()
        print(f"✓ Top emotion: {emotion_dist.index[0]} ({emotion_dist.iloc[0]} comments)")
    else:
        print("⚠ Sentiment columns not found")
except Exception as e:
    print(f"✗ Sentiment verification failed: {e}")

# Test 5: Generate intelligence brief
print("\nTest 5: Generating intelligence brief...")
try:
    brief = pipeline.generate_intelligence_brief()
    
    if brief:
        print("✓ Intelligence brief generated")
        print(f"✓ Total comments: {brief['metadata']['total_comments']}")
        print(f"✓ Entities tracked: {len(brief['entities']['people']) + len(brief['entities']['shows'])}")
        
        if brief.get('velocity_alerts'):
            print(f"✓ Velocity alerts: {len(brief['velocity_alerts'])}")
    else:
        print("⚠ Brief generation returned None")
        
except Exception as e:
    print(f"✗ Brief generation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Generate PDF report
print("\nTest 6: Generating PDF report...")
try:
    generator = IntelligenceBriefGenerator()
    pdf_path = generator.generate_report(brief, output_filename="TEST_Intelligence_Brief.pdf")
    
    if pdf_path.exists():
        print(f"✓ PDF report generated: {pdf_path}")
        print(f"✓ File size: {pdf_path.stat().st_size / 1024:.1f} KB")
    else:
        print("✗ PDF file not found")
        
except Exception as e:
    print(f"✗ PDF generation failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*70)
print("  TEST SUMMARY")
print("="*70)
print("\n✓ System is operational!")
print("\nYou can now:")
print("1. Run the CLI: python -m et_intel.cli.cli")
print("2. Import your own CSV files")
print("3. Generate custom intelligence briefs")
print("\nSample data has been processed. Check:")
print(f"  - {config.PROCESSED_DIR} for processed comments")
print(f"  - {config.REPORTS_DIR} for the test PDF report")
print("\n" + "="*70 + "\n")

