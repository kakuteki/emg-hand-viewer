"""
Hand Viewer API
===============

他のアプリケーションから呼び出し可能な手モデルビューワー

使用例:
    from emg_realtime_viz import HandViewer

    # ビューワー起動
    viewer = HandViewer()
    viewer.start()

    # 手のポーズを更新
    viewer.set_hand_pose(angles)  # 20次元の関節角度 (0-1)

    # 両手を個別に更新
    viewer.set_ground_truth(gt_angles)
    viewer.set_prediction(pred_angles)

    # 終了
    viewer.stop()
"""

from __future__ import annotations

import time
from queue import Empty, Queue
from threading import Event, Thread
from typing import Optional, Tuple

import numpy as np

# PyQt5 imports
HAS_PYQT = False
try:
    import pyqtgraph.opengl as gl
    from PyQt5 import QtWidgets
    from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal

    HAS_PYQT = True
except ImportError:
    pass

from .hand_model import DualHandModel3D


def _create_signal_emitter():
    """
    Qt シグナル用のエミッターを作る

    QObjectを継承したクラスはPyQtが無いと定義できないため、
    モジュールの読み込み時ではなく必要になった時点で組み立てる。
    こうしないとPyQtの無い環境でライブラリ全体が読み込めない。
    """
    if not HAS_PYQT:
        raise ImportError("PyQt5 and pyqtgraph are required")

    class _SignalEmitter(QObject):
        update_signal = pyqtSignal()
        close_signal = pyqtSignal()

    return _SignalEmitter()


class HandViewer:
    """
    手モデルビューワー（外部API用）

    他のアプリケーションから手のポーズをリアルタイムで可視化するためのラッパー。
    別スレッドでGUIを実行し、メインスレッドから手のポーズを更新可能。

    Parameters
    ----------
    title : str
        ウィンドウタイトル
    size : tuple
        ウィンドウサイズ (width, height)
    show_ground_truth : bool
        実測値（青い手）を表示するか
    show_prediction : bool
        予測値（緑の手）を表示するか
    angle_scale : float
        関節角度のスケール

    Examples
    --------
    >>> viewer = HandViewer()
    >>> viewer.start()
    >>>
    >>> # メインループで手のポーズを更新
    >>> while running:
    ...     angles = get_hand_pose()  # 何らかの方法で取得
    ...     viewer.set_prediction(angles)
    ...     time.sleep(0.01)
    >>>
    >>> viewer.stop()
    """

    def __init__(
        self,
        title: str = "Hand Viewer",
        size: Tuple[int, int] = (800, 600),
        show_ground_truth: bool = True,
        show_prediction: bool = True,
        angle_scale: float = 1.0,
    ):
        if not HAS_PYQT:
            raise ImportError("PyQt5 and pyqtgraph are required")

        self.title = title
        self.size = size
        self.show_ground_truth = show_ground_truth
        self.show_prediction = show_prediction
        self.angle_scale = angle_scale

        # 状態
        self._running = False
        self._thread: Optional[Thread] = None
        self._stop_event = Event()

        # 更新キュー
        self._gt_queue: Queue = Queue(maxsize=10)
        self._pred_queue: Queue = Queue(maxsize=10)

        # GUI要素（スレッド内で初期化）
        self._app = None
        self._window = None
        self._gl_widget = None
        self._hand_model: Optional[DualHandModel3D] = None
        self._timer = None
        self._emitter = None

    def start(self, blocking: bool = False):
        """
        ビューワーを開始

        Parameters
        ----------
        blocking : bool
            Trueの場合、ウィンドウが閉じるまでブロック
            Falseの場合、別スレッドで実行（デフォルト）
        """
        if self._running:
            return

        self._stop_event.clear()

        if blocking:
            self._run_gui()
        else:
            self._thread = Thread(target=self._run_gui, daemon=True)
            self._thread.start()

            # GUIが起動するまで待機
            time.sleep(0.5)

    def stop(self):
        """ビューワーを停止"""
        self._stop_event.set()

        if self._emitter:
            try:
                self._emitter.close_signal.emit()
            except:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._running = False

    def set_hand_pose(self, angles: np.ndarray, target: str = "prediction"):
        """
        手のポーズを設定

        Parameters
        ----------
        angles : np.ndarray
            関節角度 (20,) - 各値は0-1の範囲
        target : str
            "prediction" または "ground_truth"
        """
        angles = np.asarray(angles, dtype=np.float32)

        if target == "ground_truth":
            self._put_queue(self._gt_queue, angles)
        else:
            self._put_queue(self._pred_queue, angles)

    def set_ground_truth(self, angles: np.ndarray):
        """実測値（青い手）を更新"""
        self.set_hand_pose(angles, target="ground_truth")

    def set_prediction(self, angles: np.ndarray):
        """予測値（緑の手）を更新"""
        self.set_hand_pose(angles, target="prediction")

    def set_both(self, ground_truth: np.ndarray, prediction: np.ndarray):
        """両手を同時に更新"""
        self.set_ground_truth(ground_truth)
        self.set_prediction(prediction)

    def set_angle_scale(self, scale: float):
        """角度スケールを設定"""
        self.angle_scale = scale

    def is_running(self) -> bool:
        """ビューワーが実行中か"""
        return self._running and not self._stop_event.is_set()

    def _put_queue(self, queue: Queue, data: np.ndarray):
        """キューにデータを追加（古いデータは破棄）"""
        try:
            # キューが満杯なら古いデータを破棄
            while queue.full():
                try:
                    queue.get_nowait()
                except Empty:
                    break
            queue.put_nowait(data)
        except:
            pass

    def _run_gui(self):
        """GUIメインループ（別スレッドで実行）"""
        self._running = True

        # QApplication作成
        self._app = QtWidgets.QApplication.instance()
        if self._app is None:
            self._app = QtWidgets.QApplication([])

        # シグナルエミッター
        self._emitter = _create_signal_emitter()
        self._emitter.close_signal.connect(self._on_close_signal)

        # ウィンドウ作成
        self._setup_window()

        # 更新タイマー
        self._timer = QTimer()
        self._timer.timeout.connect(self._update)
        self._timer.start(16)  # ~60 FPS

        # イベントループ
        self._app.exec_()
        self._running = False

    def _setup_window(self):
        """ウィンドウをセットアップ"""
        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle(self.title)
        self._window.resize(*self.size)
        self._window.closeEvent = self._on_window_close

        # 中央ウィジェット
        central = QtWidgets.QWidget()
        self._window.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # 3Dビュー
        self._gl_widget = gl.GLViewWidget()
        self._gl_widget.setCameraPosition(distance=6, elevation=25, azimuth=45)
        layout.addWidget(self._gl_widget, stretch=3)

        # グリッド
        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        grid.translate(0, 0, -0.5)
        self._gl_widget.addItem(grid)

        # 軸
        axis = gl.GLAxisItem()
        axis.setSize(1, 1, 1)
        self._gl_widget.addItem(axis)

        # 手モデル
        self._hand_model = DualHandModel3D(self._gl_widget)
        self._hand_model.ground_truth.set_visible(self.show_ground_truth)
        self._hand_model.prediction.set_visible(self.show_prediction)

        # コントロールパネル
        control = self._create_control_panel()
        layout.addWidget(control, stretch=1)

        self._window.show()

    def _create_control_panel(self) -> QtWidgets.QWidget:
        """コントロールパネル作成"""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # タイトル
        title = QtWidgets.QLabel("Hand Viewer")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # 表示設定
        display_group = QtWidgets.QGroupBox("Display")
        display_layout = QtWidgets.QVBoxLayout(display_group)

        self._cb_gt = QtWidgets.QCheckBox("Ground Truth (Blue)")
        self._cb_gt.setChecked(self.show_ground_truth)
        self._cb_gt.toggled.connect(lambda c: self._hand_model.ground_truth.set_visible(c))
        display_layout.addWidget(self._cb_gt)

        self._cb_pred = QtWidgets.QCheckBox("Prediction (Green)")
        self._cb_pred.setChecked(self.show_prediction)
        self._cb_pred.toggled.connect(lambda c: self._hand_model.prediction.set_visible(c))
        display_layout.addWidget(self._cb_pred)

        layout.addWidget(display_group)

        # スケール
        scale_group = QtWidgets.QGroupBox("Angle Scale")
        scale_layout = QtWidgets.QVBoxLayout(scale_group)

        self._scale_slider = QtWidgets.QSlider(Qt.Horizontal)
        self._scale_slider.setRange(1, 30)
        self._scale_slider.setValue(int(self.angle_scale * 10))
        self._scale_slider.valueChanged.connect(lambda v: setattr(self, "angle_scale", v / 10))
        scale_layout.addWidget(self._scale_slider)

        self._scale_label = QtWidgets.QLabel(f"Scale: {self.angle_scale:.1f}")
        self._scale_slider.valueChanged.connect(
            lambda v: self._scale_label.setText(f"Scale: {v / 10:.1f}")
        )
        scale_layout.addWidget(self._scale_label)

        layout.addWidget(scale_group)

        # レジェンド
        legend_group = QtWidgets.QGroupBox("Legend")
        legend_layout = QtWidgets.QVBoxLayout(legend_group)
        legend_layout.addWidget(QtWidgets.QLabel("Blue: Ground Truth (Left)"))
        legend_layout.addWidget(QtWidgets.QLabel("Green: Prediction (Right)"))
        layout.addWidget(legend_group)

        layout.addStretch()

        return panel

    def _update(self):
        """更新処理"""
        if self._stop_event.is_set():
            self._timer.stop()
            self._window.close()
            return

        # Ground Truth更新
        try:
            while True:
                angles = self._gt_queue.get_nowait()
                self._hand_model.ground_truth.update_from_angles(angles, self.angle_scale)
        except Empty:
            pass

        # Prediction更新
        try:
            while True:
                angles = self._pred_queue.get_nowait()
                self._hand_model.prediction.update_from_angles(angles, self.angle_scale)
        except Empty:
            pass

    def _on_close_signal(self):
        """クローズシグナル受信時"""
        if self._timer:
            self._timer.stop()
        if self._window:
            self._window.close()
        if self._app:
            self._app.quit()

    def _on_window_close(self, event):
        """ウィンドウクローズイベント"""
        self._stop_event.set()
        if self._timer:
            self._timer.stop()
        event.accept()
        if self._app:
            self._app.quit()


class HandViewerContext:
    """
    コンテキストマネージャー対応のHandViewer

    使用例:
        with HandViewerContext() as viewer:
            for data in stream:
                viewer.set_prediction(data)
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.viewer = None

    def __enter__(self) -> HandViewer:
        self.viewer = HandViewer(**self.kwargs)
        self.viewer.start()
        return self.viewer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.viewer:
            self.viewer.stop()
        return False
