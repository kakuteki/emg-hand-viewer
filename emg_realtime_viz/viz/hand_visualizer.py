"""
Hand Visualizer with Inference Display
=======================================

EMG信号と推論結果（手のポーズ）を同時に可視化
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

# PyQt5 / PyQtGraph imports
HAS_PYQT = False
QtWidgets = None
QtCore = None
Qt = None
QTimer = None
pg = None
gl = None

try:
    import pyqtgraph as _pg
    import pyqtgraph.opengl as _gl
    from PyQt5 import QtCore as _QtCore
    from PyQt5 import QtWidgets as _QtWidgets
    from PyQt5.QtCore import Qt as _Qt
    from PyQt5.QtCore import QTimer as _QTimer

    QtWidgets = _QtWidgets
    QtCore = _QtCore
    Qt = _Qt
    QTimer = _QTimer
    pg = _pg
    gl = _gl
    HAS_PYQT = True
except ImportError:
    pass

from ..core.feature_extractor import FeatureExtractor
from ..devices.file_source import NinaproDataSource
from .hand_model import DualHandModel3D


class HandVisualizer:
    """
    手モデル付きリアルタイム可視化アプリケーション

    EMG信号の3D軌跡と、推論された手のポーズを同時に表示
    プルダウンメニューでデータセットの被験者・動作を選択可能

    Parameters
    ----------
    filepath : str
        Ninapro DB5データファイルのパス
    inference_model : callable, optional
        推論モデル
    show_ground_truth : bool
        実測値を表示するか
    show_prediction : bool
        予測値を表示するか
    """

    def __init__(
        self,
        filepath: str,
        inference_model: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        show_ground_truth: bool = True,
        show_prediction: bool = True,
        window_size: int = 20,
        update_interval: int = 10,
        playback_speed: float = 1.0,
    ):
        if not HAS_PYQT:
            raise ImportError("PyQt5 and pyqtgraph are required")

        self.filepath = Path(filepath)
        self.inference_model = inference_model
        self.show_ground_truth = show_ground_truth
        self.show_prediction = show_prediction
        self.window_size = window_size
        self.update_interval = update_interval
        self.playback_speed = playback_speed

        # データセット情報をロード
        self._load_dataset_info()

        # 現在のデータソース
        self.source: Optional[NinaproDataSource] = None
        self._current_subject = None
        self._current_movement = None

        # 特徴量抽出器（後で初期化）
        self.feature_extractor = None

        # EMG軌跡データ
        self._emg_trail = deque(maxlen=300)
        self._emg_colors = deque(maxlen=300)

        # 状態
        self._running = False
        self._paused = False
        self._data_generator = None

        # 現在の関節角度
        self._current_gt_angles: Optional[np.ndarray] = None
        self._current_pred_angles: Optional[np.ndarray] = None

        # GUI要素
        self._app = None
        self._window = None
        self._gl_widget = None
        self._emg_scatter = None
        self._hand_model: Optional[DualHandModel3D] = None
        self._timer = None

        # 統計
        self._frame_count = 0
        self._last_update_time = None

    def _load_dataset_info(self):
        """データセットの情報をロード"""
        import numpy as np

        data = np.load(self.filepath, allow_pickle=True)
        segments = data["segments"]

        # 利用可能な被験者・動作・エクササイズを収集
        self._subjects = set()
        self._movements = set()
        self._exercises = set()
        self._segment_info = []

        for seg in segments:
            if hasattr(seg, "item"):
                seg = seg.item()

            subject_id = seg.get("subject_id", 0)
            movement = seg.get("movement", 0)
            exercise_id = seg.get("exercise_id", 0)
            repetition = seg.get("repetition", 0)

            self._subjects.add(subject_id)
            self._movements.add(movement)
            self._exercises.add(exercise_id)
            self._segment_info.append(
                {
                    "subject_id": subject_id,
                    "movement": movement,
                    "exercise_id": exercise_id,
                    "repetition": repetition,
                }
            )

        self._subjects = sorted(self._subjects)
        self._movements = sorted(self._movements)
        self._exercises = sorted(self._exercises)

        print(f"Dataset loaded: {len(segments)} segments")
        print(f"  Subjects: {self._subjects}")
        print(
            f"  Movements: {min(self._movements)}-{max(self._movements)} ({len(self._movements)} types)"
        )
        print(f"  Exercises: {self._exercises}")

    def _create_source(self, subject_id: int, movement: int):
        """指定された被験者・動作のデータソースを作成"""
        if self.source is not None:
            if self.source.is_connected:
                self.source.disconnect()

        self.source = NinaproDataSource(
            filepath=str(self.filepath),
            subject_ids=[subject_id],
            movements=[movement],
            loop=True,
            n_channels=16,
            sample_rate=200.0,
        )

        # 特徴量抽出器を初期化
        self.feature_extractor = FeatureExtractor(
            window_size=self.window_size,
            n_channels=self.source.n_channels,
            features=["mav", "rms", "var", "wl"],
        )

        self._current_subject = subject_id
        self._current_movement = movement

        return self.source

    def _setup_ui(self):
        """UIセットアップ"""
        self._app = QtWidgets.QApplication.instance()
        if self._app is None:
            self._app = QtWidgets.QApplication([])

        # メインウィンドウ
        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle("EMG Hand Visualizer - Ninapro DB5")
        self._window.resize(1400, 900)

        # 中央ウィジェット
        central = QtWidgets.QWidget()
        self._window.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # 3Dビューエリア
        view_container = QtWidgets.QWidget()
        view_layout = QtWidgets.QVBoxLayout(view_container)

        # 3Dビュー
        self._gl_widget = gl.GLViewWidget()
        self._gl_widget.setCameraPosition(distance=8, elevation=30, azimuth=45)
        view_layout.addWidget(self._gl_widget)

        main_layout.addWidget(view_container, stretch=3)

        # グリッドと軸
        grid = gl.GLGridItem()
        grid.scale(3, 3, 1)
        grid.translate(0, 0, -1)
        self._gl_widget.addItem(grid)

        axis = gl.GLAxisItem()
        axis.setSize(2, 2, 2)
        self._gl_widget.addItem(axis)

        # EMG軌跡用散布図
        self._emg_scatter = gl.GLScatterPlotItem()
        self._gl_widget.addItem(self._emg_scatter)

        # 手モデル
        self._hand_model = DualHandModel3D(self._gl_widget)
        self._hand_model.ground_truth.set_visible(self.show_ground_truth)
        self._hand_model.prediction.set_visible(self.show_prediction)

        # コントロールパネル
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)

        # ステータスバー
        self._status_bar = self._window.statusBar()
        self._status_bar.showMessage("Ready - Select data and press Play")

    def _create_control_panel(self) -> QtWidgets.QWidget:
        """コントロールパネル作成"""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # タイトル
        title = QtWidgets.QLabel("Hand Visualizer")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # データ選択グループ
        data_group = QtWidgets.QGroupBox("Data Selection")
        data_layout = QtWidgets.QFormLayout(data_group)

        # 被験者選択
        self._subject_combo = QtWidgets.QComboBox()
        for s in self._subjects:
            self._subject_combo.addItem(f"Subject {s}", s)
        self._subject_combo.currentIndexChanged.connect(self._on_data_selection_changed)
        data_layout.addRow("Subject:", self._subject_combo)

        # 動作選択
        self._movement_combo = QtWidgets.QComboBox()
        for m in self._movements:
            self._movement_combo.addItem(f"Movement {m}", m)
        self._movement_combo.currentIndexChanged.connect(self._on_data_selection_changed)
        data_layout.addRow("Movement:", self._movement_combo)

        layout.addWidget(data_group)

        # 再生コントロール
        playback_group = QtWidgets.QGroupBox("Playback")
        playback_layout = QtWidgets.QHBoxLayout(playback_group)

        self._btn_play = QtWidgets.QPushButton("▶ Play")
        self._btn_play.clicked.connect(self._on_play)
        playback_layout.addWidget(self._btn_play)

        self._btn_pause = QtWidgets.QPushButton("⏸ Pause")
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_pause.setEnabled(False)
        playback_layout.addWidget(self._btn_pause)

        self._btn_reset = QtWidgets.QPushButton("⏹ Reset")
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

        # 表示設定
        display_group = QtWidgets.QGroupBox("Display")
        display_layout = QtWidgets.QVBoxLayout(display_group)

        self._cb_show_gt = QtWidgets.QCheckBox("Ground Truth (Blue)")
        self._cb_show_gt.setChecked(self.show_ground_truth)
        self._cb_show_gt.toggled.connect(self._on_toggle_gt)
        display_layout.addWidget(self._cb_show_gt)

        self._cb_show_pred = QtWidgets.QCheckBox("Prediction (Green)")
        self._cb_show_pred.setChecked(self.show_prediction)
        self._cb_show_pred.toggled.connect(self._on_toggle_pred)
        display_layout.addWidget(self._cb_show_pred)

        self._cb_show_emg = QtWidgets.QCheckBox("EMG Trajectory")
        self._cb_show_emg.setChecked(True)
        self._cb_show_emg.toggled.connect(self._on_toggle_emg)
        display_layout.addWidget(self._cb_show_emg)

        layout.addWidget(display_group)

        # 角度スケール
        scale_group = QtWidgets.QGroupBox("Angle Scale")
        scale_layout = QtWidgets.QVBoxLayout(scale_group)

        self._angle_scale_slider = QtWidgets.QSlider(Qt.Horizontal)
        self._angle_scale_slider.setRange(1, 30)
        self._angle_scale_slider.setValue(10)
        scale_layout.addWidget(self._angle_scale_slider)

        self._angle_scale_label = QtWidgets.QLabel("Scale: 1.0")
        self._angle_scale_slider.valueChanged.connect(
            lambda v: self._angle_scale_label.setText(f"Scale: {v / 10:.1f}")
        )
        scale_layout.addWidget(self._angle_scale_label)

        layout.addWidget(scale_group)

        # 情報表示
        info_group = QtWidgets.QGroupBox("Information")
        info_layout = QtWidgets.QVBoxLayout(info_group)

        self._info_labels = {}
        for key in ["Subject", "Movement", "Sample", "Segments", "FPS", "Error"]:
            label = QtWidgets.QLabel(f"{key}: --")
            self._info_labels[key] = label
            info_layout.addWidget(label)

        layout.addWidget(info_group)

        # スペーサー
        layout.addStretch()

        # レジェンド
        legend_group = QtWidgets.QGroupBox("Legend")
        legend_layout = QtWidgets.QVBoxLayout(legend_group)
        legend_layout.addWidget(QtWidgets.QLabel("🔵 Ground Truth (Left)"))
        legend_layout.addWidget(QtWidgets.QLabel("🟢 Prediction (Right)"))
        legend_layout.addWidget(QtWidgets.QLabel("🟣 EMG Trajectory (Center)"))
        layout.addWidget(legend_group)

        return panel

    def _on_data_selection_changed(self):
        """データ選択変更時"""
        # 再生中なら停止
        if self._running:
            self._on_reset()

        self._status_bar.showMessage("Data selection changed - Press Play to start")

    def _on_play(self):
        """再生"""
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
        self._running = False
        self._paused = False
        if self._timer:
            self._timer.stop()

        self._emg_trail.clear()
        self._emg_colors.clear()
        self._data_generator = None

        if self.source is not None:
            self.source.reset()
        if self.feature_extractor is not None:
            self.feature_extractor.reset()

        self._update_emg_scatter()
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._status_bar.showMessage("Reset")

    def _on_speed_change(self, value):
        """速度変更"""
        self.playback_speed = value / 10.0
        self._speed_label.setText(f"Speed: {self.playback_speed:.1f}x")

    def _on_toggle_gt(self, checked):
        """実測値表示切替"""
        self.show_ground_truth = checked
        self._hand_model.ground_truth.set_visible(checked)

    def _on_toggle_pred(self, checked):
        """予測値表示切替"""
        self.show_prediction = checked
        self._hand_model.prediction.set_visible(checked)

    def _on_toggle_emg(self, checked):
        """EMG軌跡表示切替"""
        self._emg_scatter.setVisible(checked)

    def _start_streaming(self):
        """ストリーミング開始"""
        # 選択されたデータでソースを作成
        subject_id = self._subject_combo.currentData()
        movement = self._movement_combo.currentData()

        self._create_source(subject_id, movement)

        if not self.source.connect():
            self._status_bar.showMessage("Failed to connect to data source")
            return

        # セグメント数を表示
        self._info_labels["Segments"].setText(f"Segments: {self.source.total_segments}")

        self._data_generator = self.source.stream()
        self._running = True
        self._paused = False

        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(True)

        # コンボボックスを無効化
        self._subject_combo.setEnabled(False)
        self._movement_combo.setEnabled(False)

        if self._timer is None:
            self._timer = QTimer()
            self._timer.timeout.connect(self._update)
        self._timer.start(self.update_interval)

        self._status_bar.showMessage(f"Streaming: Subject {subject_id}, Movement {movement}")
        self._last_update_time = time.time()
        self._frame_count = 0

    def _update(self):
        """更新処理"""
        if not self._running or self._paused:
            return

        if self._data_generator is None:
            return

        try:
            samples_per_update = max(1, int(self.playback_speed * 2))

            for _ in range(samples_per_update):
                emg, glove, metadata = next(self._data_generator)

                # 特徴量抽出
                self.feature_extractor.update(emg.flatten())

                if self.feature_extractor.is_ready:
                    features = self.feature_extractor.extract()

                    # EMG軌跡更新
                    self._update_emg_trail(features, metadata)

                    # 推論実行
                    if self.inference_model is not None:
                        try:
                            self._current_pred_angles = self.inference_model(features)
                        except Exception as e:
                            print(f"Inference error: {e}")
                            self._current_pred_angles = None

                    # 実測値（gloveデータ）
                    if glove is not None:
                        self._current_gt_angles = self._normalize_glove_data(glove.flatten())

            # 手モデル更新
            angle_scale = self._angle_scale_slider.value() / 10.0
            self._update_hand_models(angle_scale)

            # EMG散布図更新
            self._update_emg_scatter()

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

        except StopIteration:
            self._running = False
            self._timer.stop()
            self._btn_play.setEnabled(True)
            self._btn_pause.setEnabled(False)
            self._subject_combo.setEnabled(True)
            self._movement_combo.setEnabled(True)
            self._status_bar.showMessage("End of data")

    def _update_emg_trail(self, features: np.ndarray, metadata: Dict):
        """EMG軌跡を更新"""
        n_channels = self.source.n_channels
        n_features = len(self.feature_extractor.features)

        # 特徴量を3次元に射影
        features_reshaped = features.reshape(n_features, n_channels)
        feature_means = np.mean(features_reshaped, axis=1)

        if len(feature_means) >= 3:
            point = feature_means[:3]
            point = (point - np.mean(point)) / (np.std(point) + 1e-8)
            point[0] *= 0.5
            self._emg_trail.append(point)

    def _update_emg_scatter(self):
        """EMG散布図更新"""
        if len(self._emg_trail) == 0:
            return

        pos = np.array(self._emg_trail)
        n_points = len(pos)

        colors = np.zeros((n_points, 4))
        for i in range(n_points):
            t = i / max(1, n_points - 1)
            colors[i] = [0.5 + 0.5 * t, 0.2, 0.8 - 0.3 * t, 0.3 + 0.7 * t]

        sizes = np.linspace(3, 12, n_points)
        self._emg_scatter.setData(pos=pos, color=colors, size=sizes)

    def _normalize_glove_data(self, glove: np.ndarray) -> np.ndarray:
        """Ninapro DB5のgloveデータを0-1の範囲に正規化"""
        glove_min = np.array(
            [
                -30,
                -30,
                -10,
                -10,
                -10,
                -10,
                -10,
                -10,
                -30,
                -30,
                -10,
                -10,
                -30,
                -30,
                -10,
                -10,
                -10,
                -10,
                -10,
                -10,
                -10,
                -10,
            ]
        )

        glove_max = np.array(
            [
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                500,
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                100,
            ]
        )

        normalized = (glove - glove_min) / (glove_max - glove_min + 1e-8)
        normalized = np.clip(normalized, 0, 1)

        angles = np.zeros(20)
        angles[0:4] = normalized[0:4]
        angles[4:8] = normalized[4:8]
        angles[8:12] = normalized[8:12]
        angles[12:16] = normalized[12:16]
        angles[16:20] = normalized[16:20]

        return angles

    def _update_hand_models(self, angle_scale: float):
        """手モデル更新"""
        if self._current_gt_angles is not None and self.show_ground_truth:
            self._hand_model.ground_truth.update_from_angles(self._current_gt_angles, angle_scale)

        if self._current_pred_angles is not None and self.show_prediction:
            self._hand_model.prediction.update_from_angles(self._current_pred_angles, angle_scale)

        if self._current_gt_angles is not None and self._current_pred_angles is not None:
            error = np.mean(
                np.abs(
                    self._current_gt_angles[: len(self._current_pred_angles)]
                    - self._current_pred_angles[: len(self._current_gt_angles)]
                )
            )
            self._info_labels["Error"].setText(f"Error: {error:.4f}")

    def _update_info(self, metadata: Dict):
        """情報更新"""
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
        return self._app.exec_()

    def close(self):
        """クローズ"""
        if self._timer:
            self._timer.stop()
        if self.source and self.source.is_connected:
            self.source.disconnect()
        if self._window:
            self._window.close()
