"""
ET Social Intelligence System
Transform ET's social comment section into strategic market intelligence
"""

__version__ = "1.0.0"

# Import config for easy access
from . import config

# Setup logging on package import (after config is loaded)
try:
    from .core.logging_config import setup_logging
    setup_logging()
except ImportError:
    # Logging config may not be available in some edge cases
    pass

# Export main classes for convenience
from .core.pipeline import ETIntelligencePipeline
from .core.ingestion import CommentIngester
from .core.entity_extraction import EntityExtractor
from .core.sentiment_analysis import SentimentAnalyzer
from .reporting.report_generator import IntelligenceBriefGenerator

# Optional imports (may fail if dependencies not installed)
try:
    from .core.relationship_graph import RelationshipGraph
    __all__ = [
        'config',
        'ETIntelligencePipeline',
        'CommentIngester',
        'EntityExtractor',
        'SentimentAnalyzer',
        'RelationshipGraph',
        'IntelligenceBriefGenerator',
    ]
except ImportError:
    __all__ = [
        'config',
        'ETIntelligencePipeline',
        'CommentIngester',
        'EntityExtractor',
        'SentimentAnalyzer',
        'IntelligenceBriefGenerator',
    ]



