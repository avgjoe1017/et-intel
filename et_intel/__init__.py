"""
ET Social Intelligence System
Transform ET's social comment section into strategic market intelligence
"""

__version__ = "1.0.0"

# Import config for easy access
from . import config

# Export main classes for convenience
from .core.pipeline import ETIntelligencePipeline
from .core.ingestion import CommentIngester
from .core.entity_extraction import EntityExtractor
from .core.sentiment_analysis import SentimentAnalyzer
from .reporting.report_generator import IntelligenceBriefGenerator

__all__ = [
    'config',
    'ETIntelligencePipeline',
    'CommentIngester',
    'EntityExtractor',
    'SentimentAnalyzer',
    'IntelligenceBriefGenerator',
]



