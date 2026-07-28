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

# PyTorchはQtより先に読み込む。
# Windowsで逆の順にすると、torchのDLL初期化が失敗して
# 学習済みモデルが一切読めなくなる（WinError 1114）。
try:
    import torch  # noqa: F401
except ImportError:
    pass

from emg_realtime_viz import qt_compat

# pyqtgraphがPySide6を掴まないよう、読み込む前にQtを指定する
qt_compat.ensure()

import pyqtgraph.opengl as gl  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402
from PyQt5.QtCore import Qt, QTimer  # noqa: E402
from PyQt5.QtWidgets import QFileDialog, QMessageBox  # noqa: E402

from emg_realtime_viz.core.glove import glove_to_angles  # noqa: E402
from emg_realtime_viz.core.inference import energy_demo_model, load_torch_model  # noqa: E402
from emg_realtime_viz.viz.hand_model import DualHandModel3D  # noqa: E402

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
        self._segment_starts: List[int] = []
        self._total_frames = 0

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
        legend_layout.addWidget(QtWidgets.QLabel("Blue: Ground Truth"))
        legend_layout.addWidget(QtWidgets.QLabel("Green: Prediction"))
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
            self.inference_model = energy_demo_model()
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
            self.status_label.setText(f"Loading model: {model_path.name}...")
            QtWidgets.QApplication.processEvents()

            inference = load_torch_model(str(model_path))

            self.inference_model = inference
            self.current_model_path = model_path
            self.model_info_label.setText(f"Loaded: {model_path.name}\nDevice: {inference.device}")
            self.status_label.setText("Model loaded")

        except Exception as e:
            self.model_info_label.setText(f"Error: {str(e)[:100]}")
            self.inference_model = energy_demo_model()
            self.current_model_path = None
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

        # 進捗の計算に使う総フレーム数は、選び直したときだけ数え直す
        self._segment_starts = []
        total = 0
        for seg in self.current_segments:
            self._segment_starts.append(total)
            total += seg["emg"].shape[0]
        self._total_frames = total

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
            self.inference_model = energy_demo_model()

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
        gt_angles = glove_to_angles(glove_frame)

        # 推論（EMGの生値をそのまま渡す）
        if self.inference_model:
            try:
                pred_angles = self.inference_model(emg_frame)
            except Exception as e:
                self.playing = False
                self.play_btn.setText("Play")
                self.timer.stop()
                self.status_label.setText("Inference stopped")
                QMessageBox.warning(
                    self,
                    "Inference Error",
                    f"推論に失敗しました:\n{e}\n\nデモモデルに切り替えます。",
                )
                self.inference_model = energy_demo_model()
                return
        else:
            pred_angles = gt_angles.copy()

        # スケール取得
        scale = self.scale_slider.value() / 10.0

        # 手モデル更新
        self.hand_model.ground_truth.update_from_angles(gt_angles, scale)
        self.hand_model.prediction.update_from_angles(pred_angles, scale)

        # UI更新
        total_frames = self._total_frames
        current_total = self._segment_starts[self.current_seg_idx] + self.current_frame_idx

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

    # ========== 外から選ばせる ==========

    def select_model(self, model_path: Path):
        """
        指定したモデルを選ぶ

        一覧に無ければ一覧に足してから選ぶ。直接読み込むだけだと
        コンボの表示が "Demo Model" のまま残り、実際に動いている
        モデルと画面の表示が食い違う。
        """
        model_path = Path(model_path)
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == str(model_path):
                self.model_combo.setCurrentIndex(i)
                return

        self.available_models[model_path.stem] = model_path
        self.model_combo.addItem(model_path.stem, str(model_path))
        self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

    def select_data(self, data_path: Path):
        """指定したデータファイルを選ぶ（一覧に無ければ足す）"""
        data_path = Path(data_path)
        for i in range(self.data_combo.count()):
            if self.data_combo.itemData(i) == str(data_path):
                self.data_combo.setCurrentIndex(i)
                return

        self.available_data[data_path.stem] = data_path
        self.data_combo.addItem(data_path.stem, str(data_path))
        self.data_combo.setCurrentIndex(self.data_combo.count() - 1)

    # ========== ユーティリティ ==========

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
            window.select_model(model_path)
        else:
            print(f"モデルファイルが見つかりません: {model_path}")

    # 初期データ指定があれば選択
    if args.data:
        data_path = Path(args.data)
        if data_path.exists():
            window.select_data(data_path)
        else:
            print(f"データファイルが見つかりません: {data_path}")

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
