#!/usr/bin/env python3
"""
ET Social Intelligence - Command Line Interface
Production-ready CLI with argparse support
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from .. import config
from ..core.pipeline import ETIntelligencePipeline
from ..reporting.report_generator import IntelligenceBriefGenerator

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE) if config.LOG_FILE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_argparse():
    """Configure command-line arguments"""
    parser = argparse.ArgumentParser(
        description='ET Social Intelligence System - Transform social comments into strategic intelligence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m et_intel.cli.cli
  
  # Process a CSV file
  python -m et_intel.cli.cli --import instagram_comments.csv --platform instagram --subject "Taylor Swift"
  
  # Batch process all unprocessed CSVs in uploads folder
  python -m et_intel.cli.cli --batch
  
  # Generate intelligence brief
  python -m et_intel.cli.cli --generate --last-30-days
  
  # Run system tests
  python -m et_intel.cli.cli --test
  
  # Check version
  python -m et_intel.cli.cli --version
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--import', dest='import_csv', metavar='CSV_FILE',
                           help='Import and process a CSV file')
    mode_group.add_argument('--batch', action='store_true',
                           help='Batch process all unprocessed CSVs in uploads folder')
    mode_group.add_argument('--generate', action='store_true',
                           help='Generate intelligence brief from processed data')
    mode_group.add_argument('--test', action='store_true',
                           help='Run system tests')
    mode_group.add_argument('--stats', action='store_true',
                           help='Show database statistics')
    mode_group.add_argument('--version', action='store_true',
                           help='Show system version')
    
    # Import options
    import_group = parser.add_argument_group('import options')
    import_group.add_argument('--platform', choices=['instagram', 'youtube'],
                            help='Platform (required for --import, optional for --batch)')
    import_group.add_argument('--subject', help='Main subject/topic of the post')
    import_group.add_argument('--url', help='Post or video URL')
    
    # Batch options
    batch_group = parser.add_argument_group('batch options')
    batch_group.add_argument('--no-auto-detect', action='store_true',
                            help='Disable platform auto-detection for batch processing')
    
    # Generate options
    gen_group = parser.add_argument_group('generate options')
    gen_group.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    gen_group.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    gen_group.add_argument('--last-7-days', action='store_true')
    gen_group.add_argument('--last-30-days', action='store_true')
    gen_group.add_argument('--last-90-days', action='store_true')
    
    # Global options
    parser.add_argument('--no-api', action='store_true',
                       help='Force rule-based sentiment (no OpenAI API)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser


def main():
    """Main entry point"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.version:
        print(f"ET Social Intelligence v{config.SYSTEM_VERSION}")
        return
    
    if args.test:
        # Import test module from tests directory
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from tests import test_system
        return
    
    if args.stats:
        from ..core.ingestion import CommentIngester
        ingester = CommentIngester()
        df = ingester.load_all_processed()
        if len(df) > 0:
            print(f"Total: {len(df):,} comments")
            print(f"Platforms: {', '.join(df['platform'].unique())}")
        else:
            print("No data yet")
        return
    
    # Initialize pipeline
    use_api = not args.no_api
    pipeline = ETIntelligencePipeline(use_api=use_api)
    
    if args.batch:
        # Batch process all unprocessed CSVs
        results = pipeline.batch_process_unprocessed(
            platform=args.platform,
            auto_detect_platform=not args.no_auto_detect
        )
        
        if results['total_found'] > 0:
            print(f"\nSummary:")
            print(f"  Processed: {results['processed']}/{results['total_found']}")
            print(f"  Failed: {results['failed']}/{results['total_found']}")
            
            if results['failed'] > 0:
                print(f"\nFailed files:")
                for result in results['results']:
                    if result['status'] == 'failed':
                        print(f"  - {result['filename']}: {result.get('error', 'Unknown error')}")
        return
    
    if args.import_csv:
        if not args.platform:
            print("Error: --platform required")
            sys.exit(1)
        
        metadata = {'subject': args.subject or '', 'post_url': args.url or ''}
        df, entities = pipeline.process_new_data(args.import_csv, args.platform, metadata)
        print(f"✓ Processed {len(df)} comments")
        return
    
    if args.generate:
        start_date = None
        if args.last_7_days:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        elif args.last_30_days:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        elif args.start_date:
            start_date = args.start_date
        
        brief = pipeline.generate_intelligence_brief(start_date=start_date)
        generator = IntelligenceBriefGenerator()
        pdf = generator.generate_report(brief)
        print(f"✓ Report: {pdf}")
        return
    
    # Interactive mode if no args
    print("\nET Social Intelligence System")
    print("Run with --help for command-line options\n")


if __name__ == "__main__":
    main()



