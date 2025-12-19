#!/usr/bin/env python3
"""
EMG Inference Application
=========================

EMG信号から手のポーズを推論し、リアルタイム3D表示
Subject/Movementをプルダウンメニューで選択可能

使用方法:
    python run_inference_app.py
    python run_inference_app.py --model path/to/model.pth
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Callable

# ライブラリパス
sys.path.insert(0, str(Path(__file__).parent))

# PyQt5
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph.opengl as gl

from emg_realtime_viz.viz.hand_model import DualHandModel3D


class InferenceApp(QtWidgets.QMainWindow):
    """
    EMG推論アプリケーション

    Subject/Movementを選択してEMGデータを再生、
    推論結果と実測値を3D手モデルで表示
    """

    def __init__(
        self,
        data_path: str,
        inference_model: Optional[Callable] = None,
        parent=None
    ):
        super().__init__(parent)

        self.data_path = data_path
        self.inference_model = inference_model or self._create_demo_model()

        # データ
        self.segments: List[Dict] = []
        self.subjects: List[int] = []
        self.movements: List[int] = []
        self.current_segments: List[Dict] = []

        # 再生状態
        self.playing = False
        self.current_seg_idx = 0
        self.current_frame_idx = 0
        self.playback_speed = 0.5

        # 更新タイマー（UIより先に初期化）
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        # UI初期化
        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        """UIセットアップ"""
        self.setWindowTitle("EMG Inference Viewer")
        self.resize(1200, 800)

        # 中央ウィジェット
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # 3Dビュー
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=6, elevation=25, azimuth=45)
        layout.addWidget(self.gl_widget, stretch=3)

        # グリッド
        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        grid.translate(0, 0, -0.5)
        self.gl_widget.addItem(grid)

        # 軸
        axis = gl.GLAxisItem()
        axis.setSize(1, 1, 1)
        self.gl_widget.addItem(axis)

        # 手モデル
        self.hand_model = DualHandModel3D(self.gl_widget)

        # コントロールパネル
        control = self._create_control_panel()
        layout.addWidget(control, stretch=1)

    def _create_control_panel(self) -> QtWidgets.QWidget:
        """コントロールパネル作成"""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # タイトル
        title = QtWidgets.QLabel("EMG Inference Viewer")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # データ選択グループ
        data_group = QtWidgets.QGroupBox("Data Selection")
        data_layout = QtWidgets.QFormLayout(data_group)

        # Subject選択
        self.subject_combo = QtWidgets.QComboBox()
        self.subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        data_layout.addRow("Subject:", self.subject_combo)

        # Movement選択
        self.movement_combo = QtWidgets.QComboBox()
        self.movement_combo.currentIndexChanged.connect(self._on_movement_changed)
        data_layout.addRow("Movement:", self.movement_combo)

        # セグメント情報
        self.segment_label = QtWidgets.QLabel("Segments: -")
        data_layout.addRow(self.segment_label)

        layout.addWidget(data_group)

        # 再生コントロール
        play_group = QtWidgets.QGroupBox("Playback")
        play_layout = QtWidgets.QVBoxLayout(play_group)

        # Play/Pauseボタン
        btn_layout = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        btn_layout.addWidget(self.play_btn)

        self.reset_btn = QtWidgets.QPushButton("⟲ Reset")
        self.reset_btn.clicked.connect(self._reset_playback)
        btn_layout.addWidget(self.reset_btn)
        play_layout.addLayout(btn_layout)

        # 速度スライダー
        speed_layout = QtWidgets.QHBoxLayout()
        speed_layout.addWidget(QtWidgets.QLabel("Speed:"))
        self.speed_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 20)
        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QtWidgets.QLabel("0.5x")
        speed_layout.addWidget(self.speed_label)
        play_layout.addLayout(speed_layout)

        # 進捗バー
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        play_layout.addWidget(self.progress_bar)

        # フレーム情報
        self.frame_label = QtWidgets.QLabel("Frame: -/-")
        play_layout.addWidget(self.frame_label)

        layout.addWidget(play_group)

        # 表示設定
        display_group = QtWidgets.QGroupBox("Display")
        display_layout = QtWidgets.QVBoxLayout(display_group)

        self.gt_checkbox = QtWidgets.QCheckBox("Ground Truth (Blue)")
        self.gt_checkbox.setChecked(True)
        self.gt_checkbox.toggled.connect(
            lambda c: self.hand_model.ground_truth.set_visible(c)
        )
        display_layout.addWidget(self.gt_checkbox)

        self.pred_checkbox = QtWidgets.QCheckBox("Prediction (Green)")
        self.pred_checkbox.setChecked(True)
        self.pred_checkbox.toggled.connect(
            lambda c: self.hand_model.prediction.set_visible(c)
        )
        display_layout.addWidget(self.pred_checkbox)

        # 角度スケール
        scale_layout = QtWidgets.QHBoxLayout()
        scale_layout.addWidget(QtWidgets.QLabel("Angle Scale:"))
        self.scale_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.scale_slider.setRange(5, 30)
        self.scale_slider.setValue(10)
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        scale_layout.addWidget(self.scale_slider)
        self.scale_label = QtWidgets.QLabel("1.0")
        scale_layout.addWidget(self.scale_label)
        display_layout.addLayout(scale_layout)

        layout.addWidget(display_group)

        # レジェンド
        legend_group = QtWidgets.QGroupBox("Legend")
        legend_layout = QtWidgets.QVBoxLayout(legend_group)
        legend_layout.addWidget(QtWidgets.QLabel("🔵 Ground Truth (Left)"))
        legend_layout.addWidget(QtWidgets.QLabel("🟢 Prediction (Right)"))
        layout.addWidget(legend_group)

        # ステータス
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        return panel

    def _load_data(self):
        """データをロード"""
        self.status_label.setText("Loading data...")
        QtWidgets.QApplication.processEvents()

        try:
            data = np.load(self.data_path, allow_pickle=True)
            self.segments = list(data['segments'])

            # 利用可能なsubject/movementを取得
            self.subjects = sorted(set(seg['subject_id'] for seg in self.segments))
            self.movements = sorted(set(seg['movement'] for seg in self.segments))

            # コンボボックスを更新
            self.subject_combo.clear()
            for s in self.subjects:
                self.subject_combo.addItem(f"Subject {s}", s)

            self.movement_combo.clear()
            for m in self.movements:
                self.movement_combo.addItem(f"Movement {m}", m)

            self.status_label.setText(f"Loaded {len(self.segments)} segments")

            # 初期データを選択
            self._update_current_segments()

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load data:\n{e}")

    def _update_current_segments(self):
        """現在選択されているsubject/movementのセグメントを更新"""
        subject = self.subject_combo.currentData()
        movement = self.movement_combo.currentData()

        if subject is None or movement is None:
            return

        self.current_segments = [
            seg for seg in self.segments
            if seg['subject_id'] == subject and seg['movement'] == movement
        ]

        self.segment_label.setText(f"Segments: {len(self.current_segments)}")
        self._reset_playback()

    def _on_subject_changed(self, index):
        """Subject変更時"""
        self._update_current_segments()

    def _on_movement_changed(self, index):
        """Movement変更時"""
        self._update_current_segments()

    def _on_speed_changed(self, value):
        """速度変更時"""
        self.playback_speed = value / 10.0
        self.speed_label.setText(f"{self.playback_speed:.1f}x")

    def _on_scale_changed(self, value):
        """スケール変更時"""
        scale = value / 10.0
        self.scale_label.setText(f"{scale:.1f}")

    def _toggle_play(self):
        """再生/一時停止切り替え"""
        if not self.current_segments:
            return

        self.playing = not self.playing

        if self.playing:
            self.play_btn.setText("⏸ Pause")
            interval = int(10 / self.playback_speed)  # 100Hz base
            self.timer.start(interval)
            self.status_label.setText("Playing...")
        else:
            self.play_btn.setText("▶ Play")
            self.timer.stop()
            self.status_label.setText("Paused")

    def _reset_playback(self):
        """再生をリセット"""
        self.playing = False
        self.play_btn.setText("▶ Play")
        self.timer.stop()
        self.current_seg_idx = 0
        self.current_frame_idx = 0
        self.progress_bar.setValue(0)
        self.frame_label.setText("Frame: 0/-")
        self.status_label.setText("Ready")

    def _update_frame(self):
        """フレーム更新"""
        if not self.current_segments:
            return

        segment = self.current_segments[self.current_seg_idx]
        emg = segment['emg']      # (T, 16)
        glove = segment['glove']  # (T, 22)
        n_frames = emg.shape[0]

        # EMGデータ
        emg_frame = emg[self.current_frame_idx, :]

        # Ground Truth
        glove_frame = glove[self.current_frame_idx, :]
        gt_angles = self._normalize_glove(glove_frame)

        # 推論
        pred_angles = self.inference_model(emg_frame)

        # スケール取得
        scale = self.scale_slider.value() / 10.0

        # 手モデル更新
        self.hand_model.ground_truth.update_from_angles(gt_angles, scale)
        self.hand_model.prediction.update_from_angles(pred_angles, scale)

        # UI更新
        total_frames = sum(seg['emg'].shape[0] for seg in self.current_segments)
        current_total = sum(
            self.current_segments[i]['emg'].shape[0]
            for i in range(self.current_seg_idx)
        ) + self.current_frame_idx

        progress = int(100 * current_total / total_frames) if total_frames > 0 else 0
        self.progress_bar.setValue(progress)
        self.frame_label.setText(
            f"Seg {self.current_seg_idx + 1}/{len(self.current_segments)}, "
            f"Frame {self.current_frame_idx + 1}/{n_frames}"
        )

        # 次フレーム
        self.current_frame_idx += 1
        if self.current_frame_idx >= n_frames:
            self.current_frame_idx = 0
            self.current_seg_idx += 1
            if self.current_seg_idx >= len(self.current_segments):
                self.current_seg_idx = 0
                self.status_label.setText("Looping...")

    def _normalize_glove(self, glove: np.ndarray) -> np.ndarray:
        """Gloveデータを正規化"""
        glove_min = np.array([
            -30, -30, -10, -10,
            -20, -10, -10, -10,
            -20, -10, -10, -10,
            -20, -10, -10, -10,
            -20, -10, -10, -10,
            0, 0
        ])
        glove_max = np.array([
            100, 100, 100, 100,
            120, 100, 100, 100,
            120, 100, 100, 100,
            120, 100, 100, 100,
            120, 100, 100, 100,
            50, 50
        ])

        normalized = (glove - glove_min) / (glove_max - glove_min + 1e-8)
        normalized = np.clip(normalized, 0, 1)

        angles = np.zeros(20)
        angles[0:4] = normalized[0:4]
        angles[4:8] = normalized[4:8]
        angles[8:12] = normalized[8:12]
        angles[12:16] = normalized[12:16]
        angles[16:20] = normalized[16:20]

        return angles

    def _create_demo_model(self) -> Callable:
        """デモ用推論モデル"""
        state = {'prev': np.zeros(20)}

        def inference(emg: np.ndarray) -> np.ndarray:
            energy = np.abs(emg)
            energy = np.clip(energy / 100, 0, 1)

            angles = np.zeros(20)

            # チャンネルを指にマッピング
            for finger in range(5):
                base = finger * 4
                ch_start = finger * 3
                ch_end = min(ch_start + 3, len(energy))
                finger_energy = np.mean(energy[ch_start:ch_end]) if ch_start < len(energy) else 0

                for joint in range(4):
                    angles[base + joint] = finger_energy * (0.7 + joint * 0.1)

            # スムージング
            smoothed = 0.3 * angles + 0.7 * state['prev']
            state['prev'] = smoothed

            return np.clip(smoothed, 0, 1)

        return inference

    def closeEvent(self, event):
        """ウィンドウクローズ時"""
        self.timer.stop()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description='EMG Inference Application')
    parser.add_argument(
        '--file', '-f',
        type=str,
        default='ninapro_db5_segmented.npz',
        help='Data file path'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Trained model path (.pth)'
    )

    args = parser.parse_args()

    # データパス
    data_path = Path(args.file)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent / args.file

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    # アプリ起動
    app = QtWidgets.QApplication(sys.argv)

    # 推論モデル（オプション）
    inference_model = None
    if args.model:
        try:
            import torch
            model = torch.load(args.model)
            model.eval()

            def inference(emg):
                with torch.no_grad():
                    x = torch.tensor(emg, dtype=torch.float32).unsqueeze(0)
                    return model(x).numpy().flatten()

            inference_model = inference
            print(f"Loaded model: {args.model}")
        except Exception as e:
            print(f"Failed to load model: {e}")

    window = InferenceApp(
        data_path=str(data_path),
        inference_model=inference_model
    )
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
