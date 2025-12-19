"""
Core modules for EMG processing
"""

from .data_loader import NinaproLoader
from .feature_extractor import FeatureExtractor
from .stream import DataStream

__all__ = ['NinaproLoader', 'FeatureExtractor', 'DataStream']
