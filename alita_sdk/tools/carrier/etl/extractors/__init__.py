"""
Extractors Module - Enhanced with UI Support

Add this to your existing extractors/__init__.py file to include UI extractor.

Author: Karen Florykian
"""
from .backend_extractors import CarrierArtifactExtractor, ComparisonExtractor
from .ui_extractors import CarrierUIReportExtractor

__all__ = [
    'CarrierArtifactExtractor',
    'CarrierUIReportExtractor',
    'ComparisonExtractor'
]