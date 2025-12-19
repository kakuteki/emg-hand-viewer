"""
EMG Realtime Visualization Library
===================================

リアルタイムEMG信号の3D可視化ライブラリ

使用例:
    from emg_realtime_viz import RealtimeVisualizer, NinaproDataSource

    # データソースの作成（ファイルから）
    source = NinaproDataSource('ninapro_db5_segmented.npz')

    # 可視化アプリの起動
    app = RealtimeVisualizer(source)
    app.run()

Myo Armbandの使用:
    from emg_realtime_viz import RealtimeVisualizer, get_myo_source

    # Myoデータソース（実機またはシミュレーション）
    source = get_myo_source(simulated=False)

    # 可視化アプリの起動
    app = RealtimeVisualizer(source)
    app.run()
"""

from .core.data_loader import NinaproLoader
from .core.feature_extractor import FeatureExtractor
from .core.stream import DataStream
from .devices.base import DataSource, RealtimeDataSource
from .devices.file_source import NinaproDataSource, FilePlaybackSource
from .devices.myo_source import MyoDataSource, SimulatedMyoSource, get_myo_source
from .viz.realtime_3d import RealtimeVisualizer
from .viz.hand_model import HandModel3D, DualHandModel3D, HandSkeleton
from .viz.hand_visualizer import HandVisualizer
from .viz.hand_viewer import HandViewer, HandViewerContext

__version__ = '0.4.0'
__author__ = 'EMG Research Team'

__all__ = [
    # Core
    'NinaproLoader',
    'FeatureExtractor',
    'DataStream',
    # Devices
    'DataSource',
    'RealtimeDataSource',
    'NinaproDataSource',
    'FilePlaybackSource',
    'MyoDataSource',
    'SimulatedMyoSource',
    'get_myo_source',
    # Visualization
    'RealtimeVisualizer',
    'HandModel3D',
    'DualHandModel3D',
    'HandSkeleton',
    'HandVisualizer',
    'HandViewer',
    'HandViewerContext',
]
