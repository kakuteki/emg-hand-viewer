"""
Device abstraction layer for EMG data sources
"""

from .base import DataSource, RealtimeDataSource
from .file_source import NinaproDataSource, FilePlaybackSource
from .myo_source import MyoDataSource, SimulatedMyoSource, get_myo_source

__all__ = [
    'DataSource',
    'RealtimeDataSource',
    'NinaproDataSource',
    'FilePlaybackSource',
    'MyoDataSource',
    'SimulatedMyoSource',
    'get_myo_source',
]
