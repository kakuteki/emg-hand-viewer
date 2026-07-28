"""
Realtime 3D Visualization Module
================================

PyQtGraph + OpenGL による高速リアルタイム3D可視化
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Dict, List, Optional

import numpy as np

# PyQt5 / PyQtGraph imports
HAS_PYQT = False
QtWidgets = None
QtCore = None
QtGui = None
Qt = None
QTimer = None
pg = None
gl = None

try:
    import pyqtgraph as _pg
    import pyqtgraph.opengl as _gl
    from PyQt5 import QtCore as _QtCore
    from PyQt5 import QtGui as _QtGui
    from PyQt5 import QtWidgets as _QtWidgets
    from PyQt5.QtCore import Qt as _Qt
    from PyQt5.QtCore import QTimer as _QTimer

    QtWidgets = _QtWidgets
    QtCore = _QtCore
    QtGui = _QtGui
    Qt = _Qt
    QTimer = _QTimer
    pg = _pg
    gl = _gl
    HAS_PYQT = True
except ImportError:
    pass

from ..core.feature_extractor import FeatureExtractor
from ..devices.base import DataSource


class RealtimeVisualizer:
    """
    リアルタイム3D EMG可視化アプリケーション

    Parameters
    ----------
    source : DataSource
        データソース
    window_size : int
        特徴量計算のウィンドウサイズ
    trail_length : int
        軌跡の長さ（点数）
    update_interval : int
        更新間隔（ミリ秒）
    playback_speed : float
        再生速度

    Examples
    --------
    >>> from emg_realtime_viz import NinaproDataSource, RealtimeVisualizer
    >>> source = NinaproDataSource('ninapro_db5_segmented.npz')
    >>> viz = RealtimeVisualizer(source)
    >>> viz.run()
    """

    def __init__(
        self,
        source: DataSource,
        window_size: int = 20,
        trail_length: int = 500,
        update_interval: int = 10,
        playback_speed: float = 1.0,
        features: Optional[List[str]] = None,
    ):
        if not HAS_PYQT:
            raise ImportError(
                "PyQt5 and pyqtgraph are required. Install with: pip install PyQt5 pyqtgraph PyOpenGL"
            )

        self.source = source
        self.window_size = window_size
        self.trail_length = trail_length
        self.update_interval = update_interval
        self.playback_speed = playback_speed

        # 特徴量抽出器
        self.feature_extractor = FeatureExtractor(
            window_size=window_size,
            n_channels=source.n_channels,
            features=features or ["mav", "rms", "var"],
        )

        # 軌跡データ
        self._trail = deque(maxlen=trail_length)
        self._colors = deque(maxlen=trail_length)
        self._metadata_history = deque(maxlen=trail_length)

        # 状態
        self._running = False
        self._paused = False
        self._data_generator = None

        # GUI要素
        self._app: Optional[QtWidgets.QApplication] = None
        self._window: Optional[QtWidgets.QMainWindow] = None
        self._gl_widget: Optional[gl.GLViewWidget] = None
        self._scatter: Optional[gl.GLScatterPlotItem] = None
        self._timer: Optional[QTimer] = None

        # コールバック
        self.on_update: Optional[Callable[[np.ndarray, Dict], None]] = None

    def _setup_ui(self):
        """UIのセットアップ"""
        self._app = QtWidgets.QApplication.instance()
        if self._app is None:
            self._app = QtWidgets.QApplication([])

        # メインウィンドウ
        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle("EMG Realtime 3D Visualizer")
        self._window.resize(1200, 800)

        # 中央ウィジェット
        central_widget = QtWidgets.QWidget()
        self._window.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout(central_widget)

        # 3Dビュー
        self._gl_widget = gl.GLViewWidget()
        self._gl_widget.setCameraPosition(distance=5)
        layout.addWidget(self._gl_widget, stretch=3)

        # グリッド追加
        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        self._gl_widget.addItem(grid)

        # 軸追加
        axis = gl.GLAxisItem()
        axis.setSize(2, 2, 2)
        self._gl_widget.addItem(axis)

        # 散布図
        self._scatter = gl.GLScatterPlotItem()
        self._gl_widget.addItem(self._scatter)

        # コントロールパネル
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel, stretch=1)

        # ステータスバー
        self._status_bar = self._window.statusBar()
        self._status_bar.showMessage("Ready")

    def _create_control_panel(self) -> QtWidgets.QWidget:
        """コントロールパネルの作成"""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # タイトル
        title = QtWidgets.QLabel("Control Panel")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # 再生コントロール
        playback_group = QtWidgets.QGroupBox("Playback")
        playback_layout = QtWidgets.QHBoxLayout(playback_group)

        self._btn_play = QtWidgets.QPushButton("Play")
        self._btn_play.clicked.connect(self._on_play)
        playback_layout.addWidget(self._btn_play)

        self._btn_pause = QtWidgets.QPushButton("Pause")
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_pause.setEnabled(False)
        playback_layout.addWidget(self._btn_pause)

        self._btn_reset = QtWidgets.QPushButton("Reset")
        self._btn_reset.clicked.connect(self._on_reset)
        playback_layout.addWidget(self._btn_reset)

        layout.addWidget(playback_group)

        # 速度コントロール
        speed_group = QtWidgets.QGroupBox("Speed")
        speed_layout = QtWidgets.QVBoxLayout(speed_group)

        self._speed_slider = QtWidgets.QSlider(Qt.Horizontal)
        self._speed_slider.setRange(1, 100)
        self._speed_slider.setValue(int(self.playback_speed * 10))
        self._speed_slider.valueChanged.connect(self._on_speed_change)
        speed_layout.addWidget(self._speed_slider)

        self._speed_label = QtWidgets.QLabel(f"Speed: {self.playback_speed:.1f}x")
        speed_layout.addWidget(self._speed_label)

        layout.addWidget(speed_group)

        # 軌跡設定
        trail_group = QtWidgets.QGroupBox("Trail")
        trail_layout = QtWidgets.QVBoxLayout(trail_group)

        self._trail_slider = QtWidgets.QSlider(Qt.Horizontal)
        self._trail_slider.setRange(50, 2000)
        self._trail_slider.setValue(self.trail_length)
        self._trail_slider.valueChanged.connect(self._on_trail_change)
        trail_layout.addWidget(self._trail_slider)

        self._trail_label = QtWidgets.QLabel(f"Trail Length: {self.trail_length}")
        trail_layout.addWidget(self._trail_label)

        self._btn_clear_trail = QtWidgets.QPushButton("Clear Trail")
        self._btn_clear_trail.clicked.connect(self._clear_trail)
        trail_layout.addWidget(self._btn_clear_trail)

        layout.addWidget(trail_group)

        # 情報表示
        info_group = QtWidgets.QGroupBox("Information")
        info_layout = QtWidgets.QVBoxLayout(info_group)

        self._info_labels = {}
        for key in ["Subject", "Movement", "Sample", "FPS"]:
            label = QtWidgets.QLabel(f"{key}: --")
            self._info_labels[key] = label
            info_layout.addWidget(label)

        layout.addWidget(info_group)

        # 色設定
        color_group = QtWidgets.QGroupBox("Color Mode")
        color_layout = QtWidgets.QVBoxLayout(color_group)

        self._color_combo = QtWidgets.QComboBox()
        self._color_combo.addItems(["Time (gradient)", "Movement", "Subject", "Energy"])
        self._color_combo.currentTextChanged.connect(self._on_color_mode_change)
        color_layout.addWidget(self._color_combo)

        layout.addWidget(color_group)

        # スペーサー
        layout.addStretch()

        return panel

    def _on_play(self):
        """再生開始"""
        if not self._running:
            self._start_streaming()
        elif self._paused:
            self._paused = False
            self._btn_play.setEnabled(False)
            self._btn_pause.setEnabled(True)

    def _on_pause(self):
        """一時停止"""
        self._paused = True
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)

    def _on_reset(self):
        """リセット"""
        self._clear_trail()
        self.source.reset()
        self.feature_extractor.reset()
        self._data_generator = None
        self._status_bar.showMessage("Reset")

    def _on_speed_change(self, value):
        """速度変更"""
        self.playback_speed = value / 10.0
        self._speed_label.setText(f"Speed: {self.playback_speed:.1f}x")

    def _on_trail_change(self, value):
        """軌跡長変更"""
        self.trail_length = value
        self._trail = deque(maxlen=value)
        self._colors = deque(maxlen=value)
        self._metadata_history = deque(maxlen=value)
        self._trail_label.setText(f"Trail Length: {value}")

    def _on_color_mode_change(self, mode):
        """色モード変更"""
        self._update_colors()

    def _clear_trail(self):
        """軌跡クリア"""
        self._trail.clear()
        self._colors.clear()
        self._metadata_history.clear()
        self._update_scatter()

    def _start_streaming(self):
        """ストリーミング開始"""
        if not self.source.is_connected:
            self.source.connect()

        self._data_generator = self.source.stream()
        self._running = True
        self._paused = False

        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(True)

        # タイマー開始
        if self._timer is None:
            self._timer = QTimer()
            self._timer.timeout.connect(self._update)
        self._timer.start(self.update_interval)

        self._status_bar.showMessage("Streaming...")
        self._last_update_time = time.time()
        self._frame_count = 0

    def _update(self):
        """更新処理（タイマーから呼び出し）"""
        if not self._running or self._paused:
            return

        if self._data_generator is None:
            return

        try:
            # 速度に応じて複数サンプル処理
            samples_per_update = max(1, int(self.playback_speed * 2))

            for _ in range(samples_per_update):
                emg, glove, metadata = next(self._data_generator)

                # 特徴量抽出
                self.feature_extractor.update(emg.flatten())

                if self.feature_extractor.is_ready:
                    features = self.feature_extractor.extract()

                    # 3次元に射影（最初の3つの特徴量を使用）
                    if len(features) >= 3:
                        point = self._project_to_3d(features)
                        self._trail.append(point)
                        self._metadata_history.append(metadata)

            # 色更新
            self._update_colors()

            # 散布図更新
            self._update_scatter()

            # 情報更新
            self._update_info(metadata)

            # FPS計算
            self._frame_count += 1
            elapsed = time.time() - self._last_update_time
            if elapsed >= 1.0:
                fps = self._frame_count / elapsed
                self._info_labels["FPS"].setText(f"FPS: {fps:.1f}")
                self._frame_count = 0
                self._last_update_time = time.time()

            # コールバック
            if self.on_update and len(self._trail) > 0:
                self.on_update(np.array(self._trail), metadata)

        except StopIteration:
            self._running = False
            self._timer.stop()
            self._btn_play.setEnabled(True)
            self._btn_pause.setEnabled(False)
            self._status_bar.showMessage("End of data")

    def _project_to_3d(self, features: np.ndarray) -> np.ndarray:
        """特徴量を3次元空間に射影"""
        # シンプルなPCA的射影（最初の3つの主要特徴量）
        # 実際にはより洗練された次元削減を使用可能

        n_channels = self.source.n_channels
        n_features = len(self.feature_extractor.features)

        # 各特徴量タイプの平均を取る
        features_reshaped = features.reshape(n_features, n_channels)
        feature_means = np.mean(features_reshaped, axis=1)

        # 正規化
        point = feature_means[:3]
        point = (point - np.mean(point)) / (np.std(point) + 1e-8)

        return point

    def _update_colors(self):
        """色の更新"""
        n_points = len(self._trail)
        if n_points == 0:
            return

        mode = (
            self._color_combo.currentText() if hasattr(self, "_color_combo") else "Time (gradient)"
        )

        if mode == "Time (gradient)":
            # 時間に基づくグラデーション
            colors = np.zeros((n_points, 4))
            for i in range(n_points):
                t = i / max(1, n_points - 1)
                colors[i] = [0.2 + 0.8 * t, 0.2, 1.0 - 0.8 * t, 0.3 + 0.7 * t]

        elif mode == "Movement":
            # 動作に基づく色
            colors = np.zeros((n_points, 4))
            cmap = [
                [0.12, 0.47, 0.71, 1],
                [0.68, 0.78, 0.91, 1],
                [1.0, 0.50, 0.05, 1],
                [1.0, 0.73, 0.47, 1],
                [0.17, 0.63, 0.17, 1],
                [0.60, 0.87, 0.54, 1],
                [0.84, 0.15, 0.16, 1],
                [1.0, 0.60, 0.59, 1],
                [0.58, 0.40, 0.74, 1],
                [0.77, 0.69, 0.84, 1],
            ]
            for i, meta in enumerate(self._metadata_history):
                if meta and "movement" in meta:
                    c_idx = meta["movement"] % len(cmap)
                    colors[i] = cmap[c_idx]
                else:
                    colors[i] = [0.5, 0.5, 0.5, 0.5]

        elif mode == "Subject":
            # 被験者に基づく色
            colors = np.zeros((n_points, 4))
            cmap = [
                [0.9, 0.1, 0.1, 1],
                [0.1, 0.9, 0.1, 1],
                [0.1, 0.1, 0.9, 1],
                [0.9, 0.9, 0.1, 1],
                [0.9, 0.1, 0.9, 1],
                [0.1, 0.9, 0.9, 1],
                [0.9, 0.5, 0.1, 1],
                [0.5, 0.1, 0.9, 1],
                [0.1, 0.5, 0.9, 1],
                [0.5, 0.9, 0.1, 1],
            ]
            for i, meta in enumerate(self._metadata_history):
                if meta and "subject_id" in meta:
                    c_idx = (meta["subject_id"] - 1) % len(cmap)
                    colors[i] = cmap[c_idx]
                else:
                    colors[i] = [0.5, 0.5, 0.5, 0.5]

        elif mode == "Energy":
            # エネルギー（点の大きさ）に基づく色
            colors = np.zeros((n_points, 4))
            trail_array = np.array(self._trail)
            energies = np.linalg.norm(trail_array, axis=1)
            energies_norm = (energies - energies.min()) / (energies.max() - energies.min() + 1e-8)
            for i, e in enumerate(energies_norm):
                colors[i] = [e, 0.3, 1 - e, 0.5 + 0.5 * e]

        else:
            colors = np.ones((n_points, 4)) * [0.5, 0.5, 1.0, 0.8]

        self._colors = deque(colors.tolist(), maxlen=self.trail_length)

    def _update_scatter(self):
        """散布図の更新"""
        if len(self._trail) == 0:
            return

        pos = np.array(self._trail)
        colors = np.array(list(self._colors))

        # サイズ（新しい点ほど大きく）
        n_points = len(self._trail)
        sizes = np.linspace(3, 15, n_points)

        self._scatter.setData(pos=pos, color=colors, size=sizes)

    def _update_info(self, metadata: Dict):
        """情報パネル更新"""
        if metadata:
            if "subject_id" in metadata:
                self._info_labels["Subject"].setText(f"Subject: {metadata['subject_id']}")
            if "movement" in metadata:
                self._info_labels["Movement"].setText(f"Movement: {metadata['movement']}")
            if "sample_idx" in metadata:
                total = metadata.get("total_samples", "?")
                self._info_labels["Sample"].setText(f"Sample: {metadata['sample_idx']}/{total}")

    def run(self):
        """アプリケーション実行"""
        self._setup_ui()
        self._window.show()

        # 初期接続
        if not self.source.is_connected:
            self.source.connect()

        return self._app.exec_()

    def close(self):
        """クローズ"""
        if self._timer:
            self._timer.stop()
        if self.source.is_connected:
            self.source.disconnect()
        if self._window:
            self._window.close()
