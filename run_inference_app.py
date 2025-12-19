#!/usr/bin/env python3
"""
EMG Inference Application
=========================

EMG信号から手のポーズを推論し、リアルタイム3D表示
- Subject/Movementをプルダウンメニューで選択
- 推論モデルをプルダウンメニューで選択
- データファイルをプルダウンメニューで選択
- モデル/データのインポート機能

使用方法:
    python run_inference_app.py

ディレクトリ構成:
    models/  - 学習済みモデル (.pth) を配置
    data/    - データファイル (.npz) を配置
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# ライブラリパス
sys.path.insert(0, str(Path(__file__).parent))

# PyQt5
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from emg_realtime_viz.viz.hand_model import DualHandModel3D

# デフォルトディレクトリ
APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"


class InferenceApp(QtWidgets.QMainWindow):
    """
    EMG推論アプリケーション

    機能:
    - Subject/Movementを選択してEMGデータを再生
    - 推論モデルを選択
    - データファイルを選択
    - モデル/データのインポート
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # ディレクトリ作成
        MODELS_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        # モデル/データ
        self.available_models: Dict[str, Path] = {}
        self.available_data: Dict[str, Path] = {}
        self.current_model_path: Optional[Path] = None
        self.current_data_path: Optional[Path] = None
        self.inference_model: Optional[Callable] = None

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

        # 更新タイマー
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        # UI初期化
        self._setup_ui()

        # モデル/データをスキャン
        self._scan_models()
        self._scan_data()

    def _setup_ui(self):
        """UIセットアップ"""
        self.setWindowTitle("EMG Inference Viewer")
        self.resize(1400, 900)

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
        panel.setMinimumWidth(350)
        layout = QtWidgets.QVBoxLayout(panel)

        # タイトル
        title = QtWidgets.QLabel("EMG Inference Viewer")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # === モデル選択グループ ===
        model_group = QtWidgets.QGroupBox("Model")
        model_layout = QtWidgets.QVBoxLayout(model_group)

        # モデル選択コンボボックス
        model_select_layout = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_select_layout.addWidget(self.model_combo, stretch=1)

        # モデルインポートボタン
        self.import_model_btn = QtWidgets.QPushButton("Import...")
        self.import_model_btn.clicked.connect(self._import_model)
        model_select_layout.addWidget(self.import_model_btn)
        model_layout.addLayout(model_select_layout)

        # モデル情報
        self.model_info_label = QtWidgets.QLabel("No model loaded")
        self.model_info_label.setStyleSheet("color: gray; font-size: 11px;")
        self.model_info_label.setWordWrap(True)
        model_layout.addWidget(self.model_info_label)

        layout.addWidget(model_group)

        # === データ選択グループ ===
        data_group = QtWidgets.QGroupBox("Data File")
        data_layout = QtWidgets.QVBoxLayout(data_group)

        # データファイル選択コンボボックス
        data_select_layout = QtWidgets.QHBoxLayout()
        self.data_combo = QtWidgets.QComboBox()
        self.data_combo.currentIndexChanged.connect(self._on_data_changed)
        data_select_layout.addWidget(self.data_combo, stretch=1)

        # データインポートボタン
        self.import_data_btn = QtWidgets.QPushButton("Import...")
        self.import_data_btn.clicked.connect(self._import_data)
        data_select_layout.addWidget(self.import_data_btn)
        data_layout.addLayout(data_select_layout)

        # データ情報
        self.data_info_label = QtWidgets.QLabel("No data loaded")
        self.data_info_label.setStyleSheet("color: gray; font-size: 11px;")
        self.data_info_label.setWordWrap(True)
        data_layout.addWidget(self.data_info_label)

        layout.addWidget(data_group)

        # === セグメント選択グループ ===
        segment_group = QtWidgets.QGroupBox("Segment Selection")
        segment_layout = QtWidgets.QFormLayout(segment_group)

        # Subject選択
        self.subject_combo = QtWidgets.QComboBox()
        self.subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        segment_layout.addRow("Subject:", self.subject_combo)

        # Movement選択
        self.movement_combo = QtWidgets.QComboBox()
        self.movement_combo.currentIndexChanged.connect(self._on_movement_changed)
        segment_layout.addRow("Movement:", self.movement_combo)

        # セグメント情報
        self.segment_label = QtWidgets.QLabel("Segments: -")
        segment_layout.addRow(self.segment_label)

        layout.addWidget(segment_group)

        # === 再生コントロール ===
        play_group = QtWidgets.QGroupBox("Playback")
        play_layout = QtWidgets.QVBoxLayout(play_group)

        # Play/Pauseボタン
        btn_layout = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        btn_layout.addWidget(self.play_btn)

        self.reset_btn = QtWidgets.QPushButton("Reset")
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

        # === 表示設定 ===
        display_group = QtWidgets.QGroupBox("Display")
        display_layout = QtWidgets.QVBoxLayout(display_group)

        self.gt_checkbox = QtWidgets.QCheckBox("Ground Truth (Blue)")
        self.gt_checkbox.setChecked(True)
        self.gt_checkbox.toggled.connect(lambda c: self.hand_model.ground_truth.set_visible(c))
        display_layout.addWidget(self.gt_checkbox)

        self.pred_checkbox = QtWidgets.QCheckBox("Prediction (Green)")
        self.pred_checkbox.setChecked(True)
        self.pred_checkbox.toggled.connect(lambda c: self.hand_model.prediction.set_visible(c))
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

        # === 凡例 ===
        legend_group = QtWidgets.QGroupBox("Legend")
        legend_layout = QtWidgets.QVBoxLayout(legend_group)
        legend_layout.addWidget(QtWidgets.QLabel("Blue: Ground Truth (Left)"))
        legend_layout.addWidget(QtWidgets.QLabel("Green: Prediction (Right)"))
        layout.addWidget(legend_group)

        # ステータス
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        return panel

    # ========== モデル管理 ==========

    def _scan_models(self):
        """modelsディレクトリをスキャン"""
        self.available_models.clear()
        self.model_combo.clear()

        # デモモデル（常に利用可能）
        self.model_combo.addItem("Demo Model (Built-in)", "demo")

        # modelsディレクトリ内の.pthファイルをスキャン
        if MODELS_DIR.exists():
            for pth_file in sorted(MODELS_DIR.glob("*.pth")):
                name = pth_file.stem
                self.available_models[name] = pth_file
                self.model_combo.addItem(name, str(pth_file))

        # ルートディレクトリの.pthファイルもスキャン
        for pth_file in sorted(APP_DIR.glob("*.pth")):
            name = pth_file.stem
            if name not in self.available_models:
                self.available_models[name] = pth_file
                self.model_combo.addItem(f"{name} (root)", str(pth_file))

    def _on_model_changed(self, index):
        """モデル選択変更時"""
        if index < 0:
            return

        model_data = self.model_combo.currentData()

        if model_data == "demo":
            self.inference_model = self._create_demo_model()
            self.current_model_path = None
            self.model_info_label.setText("Demo model: EMG energy-based estimation")
        else:
            model_path = Path(model_data)
            if model_path.exists():
                self._load_pytorch_model(model_path)
            else:
                self.model_info_label.setText("Error: File not found")

    def _load_pytorch_model(self, model_path: Path):
        """PyTorchモデルをロード"""
        try:
            import torch

            self.status_label.setText(f"Loading model: {model_path.name}...")
            QtWidgets.QApplication.processEvents()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # モデルをロード
            model = torch.load(str(model_path), map_location=device)
            model.eval()

            # 推論関数を作成
            state = {"prev": None}

            def inference(emg: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    if emg.ndim > 1:
                        emg = emg.mean(axis=0) if emg.shape[1] == 16 else emg.mean(axis=1)

                    x = torch.tensor(emg, dtype=torch.float32).unsqueeze(0).to(device)
                    y = model(x).cpu().numpy().flatten()

                    # 0-1にクリップ
                    y = np.clip(y, 0, 1)

                    # 出力が22次元の場合は20次元に変換
                    if len(y) == 22:
                        angles = np.zeros(20)
                        angles[0:4] = y[0:4]
                        angles[4:8] = y[4:8]
                        angles[8:12] = y[8:12]
                        angles[12:16] = y[12:16]
                        angles[16:20] = y[16:20]
                        y = angles

                    # スムージング
                    if state["prev"] is not None:
                        y = 0.3 * y + 0.7 * state["prev"]
                    state["prev"] = y

                    return y

            self.inference_model = inference
            self.current_model_path = model_path
            self.model_info_label.setText(f"Loaded: {model_path.name}\nDevice: {device}")
            self.status_label.setText("Model loaded")

        except Exception as e:
            self.model_info_label.setText(f"Error: {str(e)[:100]}")
            self.inference_model = self._create_demo_model()
            QMessageBox.warning(
                self, "Model Load Error", f"Failed to load model:\n{e}\n\nUsing demo model."
            )

    def _import_model(self):
        """モデルをインポート"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Model", str(Path.home()), "PyTorch Models (*.pth *.pt);;All Files (*)"
        )

        if file_path:
            src_path = Path(file_path)
            dst_path = MODELS_DIR / src_path.name

            # コピー確認
            if dst_path.exists():
                reply = QMessageBox.question(
                    self,
                    "Confirm Overwrite",
                    f"'{src_path.name}' already exists.\nOverwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

            try:
                shutil.copy2(src_path, dst_path)
                self._scan_models()

                # 新しくインポートしたモデルを選択
                for i in range(self.model_combo.count()):
                    if self.model_combo.itemData(i) == str(dst_path):
                        self.model_combo.setCurrentIndex(i)
                        break

                self.status_label.setText(f"Imported: {src_path.name}")

            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import:\n{e}")

    # ========== データ管理 ==========

    def _scan_data(self):
        """dataディレクトリをスキャン"""
        self.available_data.clear()
        self.data_combo.clear()

        # dataディレクトリ内の.npzファイルをスキャン
        if DATA_DIR.exists():
            for npz_file in sorted(DATA_DIR.glob("*.npz")):
                name = npz_file.stem
                self.available_data[name] = npz_file
                self.data_combo.addItem(name, str(npz_file))

        # ルートディレクトリの.npzファイルもスキャン
        for npz_file in sorted(APP_DIR.glob("*.npz")):
            name = npz_file.stem
            if name not in self.available_data:
                self.available_data[name] = npz_file
                self.data_combo.addItem(f"{name} (root)", str(npz_file))

        if self.data_combo.count() == 0:
            self.data_combo.addItem("No data files found", None)
            self.data_info_label.setText("Place .npz files in 'data/' directory")

    def _on_data_changed(self, index):
        """データファイル選択変更時"""
        if index < 0:
            return

        data_path = self.data_combo.currentData()

        if data_path is None:
            return

        data_path = Path(data_path)
        if data_path.exists():
            self._load_data(data_path)

    def _load_data(self, data_path: Path):
        """データをロード"""
        self.status_label.setText(f"Loading: {data_path.name}...")
        QtWidgets.QApplication.processEvents()

        try:
            data = np.load(str(data_path), allow_pickle=True)
            self.segments = list(data["segments"])
            self.current_data_path = data_path

            # 利用可能なsubject/movementを取得
            self.subjects = sorted(set(seg["subject_id"] for seg in self.segments))
            self.movements = sorted(set(seg["movement"] for seg in self.segments))

            # コンボボックスを更新
            self.subject_combo.blockSignals(True)
            self.movement_combo.blockSignals(True)

            self.subject_combo.clear()
            for s in self.subjects:
                self.subject_combo.addItem(f"Subject {s}", s)

            self.movement_combo.clear()
            for m in self.movements:
                self.movement_combo.addItem(f"Movement {m}", m)

            self.subject_combo.blockSignals(False)
            self.movement_combo.blockSignals(False)

            # 情報更新
            self.data_info_label.setText(
                f"Loaded: {data_path.name}\n"
                f"Segments: {len(self.segments)}, "
                f"Subjects: {len(self.subjects)}, "
                f"Movements: {len(self.movements)}"
            )

            self.status_label.setText("Data loaded")

            # 初期データを選択
            self._update_current_segments()

        except Exception as e:
            self.data_info_label.setText(f"Error: {str(e)[:100]}")
            QMessageBox.critical(self, "Data Load Error", f"Failed to load data:\n{e}")

    def _import_data(self):
        """データをインポート"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Data", str(Path.home()), "NumPy Files (*.npz *.npy);;All Files (*)"
        )

        if file_path:
            src_path = Path(file_path)
            dst_path = DATA_DIR / src_path.name

            # コピー確認
            if dst_path.exists():
                reply = QMessageBox.question(
                    self,
                    "Confirm Overwrite",
                    f"'{src_path.name}' already exists.\nOverwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

            try:
                shutil.copy2(src_path, dst_path)
                self._scan_data()

                # 新しくインポートしたデータを選択
                for i in range(self.data_combo.count()):
                    if self.data_combo.itemData(i) == str(dst_path):
                        self.data_combo.setCurrentIndex(i)
                        break

                self.status_label.setText(f"Imported: {src_path.name}")

            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import:\n{e}")

    # ========== セグメント管理 ==========

    def _update_current_segments(self):
        """現在選択されているsubject/movementのセグメントを更新"""
        subject = self.subject_combo.currentData()
        movement = self.movement_combo.currentData()

        if subject is None or movement is None:
            return

        self.current_segments = [
            seg
            for seg in self.segments
            if seg["subject_id"] == subject and seg["movement"] == movement
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

    # ========== 再生制御 ==========

    def _toggle_play(self):
        """再生/一時停止切り替え"""
        if not self.current_segments:
            QMessageBox.information(self, "No Data", "Please load data first.")
            return

        if self.inference_model is None:
            self.inference_model = self._create_demo_model()

        self.playing = not self.playing

        if self.playing:
            self.play_btn.setText("Pause")
            interval = int(10 / self.playback_speed)
            self.timer.start(interval)
            self.status_label.setText("Playing...")
        else:
            self.play_btn.setText("Play")
            self.timer.stop()
            self.status_label.setText("Paused")

    def _reset_playback(self):
        """再生をリセット"""
        self.playing = False
        self.play_btn.setText("Play")
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
        emg = segment["emg"]  # (T, 16)
        glove = segment["glove"]  # (T, 22)
        n_frames = emg.shape[0]

        # EMGデータ
        emg_frame = emg[self.current_frame_idx, :]

        # Ground Truth
        glove_frame = glove[self.current_frame_idx, :]
        gt_angles = self._normalize_glove(glove_frame)

        # 推論
        if self.inference_model:
            pred_angles = self.inference_model(emg_frame)
        else:
            pred_angles = gt_angles.copy()

        # スケール取得
        scale = self.scale_slider.value() / 10.0

        # 手モデル更新
        self.hand_model.ground_truth.update_from_angles(gt_angles, scale)
        self.hand_model.prediction.update_from_angles(pred_angles, scale)

        # UI更新
        total_frames = sum(seg["emg"].shape[0] for seg in self.current_segments)
        current_total = (
            sum(self.current_segments[i]["emg"].shape[0] for i in range(self.current_seg_idx))
            + self.current_frame_idx
        )

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

    # ========== ユーティリティ ==========

    def _normalize_glove(self, glove: np.ndarray) -> np.ndarray:
        """Gloveデータを正規化"""
        glove_min = np.array(
            [
                -30,
                -30,
                -10,
                -10,
                -20,
                -10,
                -10,
                -10,
                -20,
                -10,
                -10,
                -10,
                -20,
                -10,
                -10,
                -10,
                -20,
                -10,
                -10,
                -10,
                0,
                0,
            ]
        )
        glove_max = np.array(
            [
                100,
                100,
                100,
                100,
                120,
                100,
                100,
                100,
                120,
                100,
                100,
                100,
                120,
                100,
                100,
                100,
                120,
                100,
                100,
                100,
                50,
                50,
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

    def _create_demo_model(self) -> Callable:
        """デモ用推論モデル"""
        state = {"prev": np.zeros(20)}

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
            smoothed = 0.3 * angles + 0.7 * state["prev"]
            state["prev"] = smoothed

            return np.clip(smoothed, 0, 1)

        return inference

    def closeEvent(self, event):
        """ウィンドウクローズ時"""
        self.timer.stop()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="EMG Inference Application")
    parser.add_argument("--model", "-m", type=str, help="Initial model path")
    parser.add_argument("--data", "-d", type=str, help="Initial data path")

    args = parser.parse_args()

    # アプリ起動
    app = QtWidgets.QApplication(sys.argv)

    window = InferenceApp()

    # 初期モデル指定があれば選択
    if args.model:
        model_path = Path(args.model)
        if model_path.exists():
            # コンボボックスで該当モデルを探す
            for i in range(window.model_combo.count()):
                if window.model_combo.itemData(i) == str(model_path):
                    window.model_combo.setCurrentIndex(i)
                    break
            else:
                # リストにない場合は直接ロード
                window._load_pytorch_model(model_path)

    # 初期データ指定があれば選択
    if args.data:
        data_path = Path(args.data)
        if data_path.exists():
            for i in range(window.data_combo.count()):
                if window.data_combo.itemData(i) == str(data_path):
                    window.data_combo.setCurrentIndex(i)
                    break

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
