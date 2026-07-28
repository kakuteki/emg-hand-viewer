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

from . import qt_compat  # noqa: F401  pyqtgraphより先に読むこと
from .core.data_loader import NinaproLoader
from .core.feature_extractor import FeatureExtractor
from .core.glove import glove_to_angles, normalize_glove, to_hand_angles
from .core.inference import TorchInference, energy_demo_model, load_torch_model, wave_demo_model
from .core.stream import DataStream
from .devices.base import DataSource, RealtimeDataSource
from .devices.file_source import FilePlaybackSource, NinaproDataSource
from .devices.myo_source import MyoDataSource, SimulatedMyoSource, get_myo_source
from .viz.hand_model import DualHandModel3D, HandModel3D, HandSkeleton, forward_kinematics
from .viz.hand_viewer import HandViewer, HandViewerContext
from .viz.hand_visualizer import HandVisualizer
from .viz.realtime_3d import RealtimeVisualizer

__version__ = "0.5.0"
__author__ = "EMG Research Team"

__all__ = [
    # Core
    "NinaproLoader",
    "FeatureExtractor",
    "DataStream",
    # Glove / Inference
    "normalize_glove",
    "glove_to_angles",
    "to_hand_angles",
    "TorchInference",
    "load_torch_model",
    "energy_demo_model",
    "wave_demo_model",
    # Devices
    "DataSource",
    "RealtimeDataSource",
    "NinaproDataSource",
    "FilePlaybackSource",
    "MyoDataSource",
    "SimulatedMyoSource",
    "get_myo_source",
    # Visualization
    "RealtimeVisualizer",
    "HandModel3D",
    "DualHandModel3D",
    "HandSkeleton",
    "forward_kinematics",
    "HandVisualizer",
    "HandViewer",
    "HandViewerContext",
]
