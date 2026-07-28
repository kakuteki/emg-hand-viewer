"""
Hand Viewer API
===============

他のアプリケーションから呼び出せる手モデルビューワー

QtのGUIは主スレッドでしか作れない。以前は start() が別スレッドで
QApplicationとウィンドウを作っていたため、次の警告のあとプロセスごと
落ちていた。

    QApplication was not created in the main() thread.
    Cannot make QOpenGLContext current in a different thread

そのため、主スレッドで動かす2通りの使い方を用意している。

1. 自分のループの中で回す（READMEの例と同じ形）

    from emg_realtime_viz import HandViewer

    viewer = HandViewer()
    viewer.open()
    while viewer.process():
        angles = model.predict(get_emg())
        viewer.set_prediction(angles)   # ここで描画も進む
    viewer.close()

2. イベントループに任せ、データ供給を別スレッドにする

    def producer(viewer):
        while viewer.is_running():
            viewer.set_prediction(model.predict(get_emg()))

    HandViewer().run(producer)
"""

from __future__ import annotations

import threading
from queue import Empty, Queue
from threading import Event, Thread
from typing import Callable, Optional, Tuple

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
        close_signal = pyqtSignal()

    return _SignalEmitter()


def _on_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


class HandViewer:
    """
    手モデルビューワー（外部API用）

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

    Notes
    -----
    open() と run() は主スレッドから呼ぶこと。
    姿勢の更新（set_prediction など）はどのスレッドからでも呼べる。
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
        self._stop_event = Event()
        self._producer_thread: Optional[Thread] = None

        # 更新キュー（別スレッドからの受け口）
        self._gt_queue: Queue = Queue(maxsize=10)
        self._pred_queue: Queue = Queue(maxsize=10)

        # GUI要素
        self._app = None
        self._window = None
        self._gl_widget = None
        self._hand_model: Optional[DualHandModel3D] = None
        self._timer = None
        self._emitter = None

    # ------------------------------------------------------------------
    # 起動と終了
    # ------------------------------------------------------------------

    def open(self):
        """
        ウィンドウを作る（イベントループは回さない）

        主スレッドから呼ぶこと。以後 process() を呼ぶか、
        姿勢を更新するたびに描画が進む。
        """
        if self._running:
            return

        if not _on_main_thread():
            raise RuntimeError(
                "HandViewer.open() は主スレッドから呼んでください。"
                "QtのGUIは主スレッドでしか作れません。"
                "データ供給を別スレッドにしたい場合は run(producer) を使ってください。"
            )

        self._stop_event.clear()

        self._app = QtWidgets.QApplication.instance()
        if self._app is None:
            self._app = QtWidgets.QApplication([])

        self._emitter = _create_signal_emitter()
        self._emitter.close_signal.connect(self._on_close_signal)

        self._setup_window()

        self._timer = QTimer()
        self._timer.timeout.connect(self._drain)
        self._timer.start(16)  # ~60 FPS

        self._running = True
        self._app.processEvents()

    def process(self) -> bool:
        """
        溜まった描画と入力を処理する

        Returns
        -------
        bool
            ウィンドウがまだ開いていれば True
        """
        if not self._running:
            return False

        self._drain()
        self._app.processEvents()
        return self.is_running()

    def run(self, producer: Optional[Callable[[HandViewer], None]] = None):
        """
        イベントループを回す（ウィンドウが閉じるまで戻らない）

        Parameters
        ----------
        producer : callable, optional
            別スレッドで実行する関数。引数にこのビューワーが渡される。
            この中から set_prediction などを呼んで姿勢を更新する。
        """
        self.open()

        if producer is not None:
            self._producer_thread = Thread(target=self._run_producer, args=(producer,), daemon=True)
            self._producer_thread.start()

        try:
            self._app.exec_()
        finally:
            self._running = False
            self._stop_event.set()
            if self._producer_thread is not None:
                self._producer_thread.join(timeout=2.0)
                self._producer_thread = None

    def _run_producer(self, producer: Callable[[HandViewer], None]):
        try:
            producer(self)
        except Exception as e:
            print(f"Producer error: {e}")
        finally:
            self.stop()

    def start(self, blocking: bool = True):
        """
        run() の別名（後方互換）

        Parameters
        ----------
        blocking : bool
            True のときだけ対応。False は以前あった「別スレッドでGUIを回す」
            動作だが、Qtの制約により動かないため例外にしている。
        """
        if not blocking:
            raise RuntimeError(
                "別スレッドでGUIを回すことはQtの制約でできません。"
                "自分のループの中で回すなら open() と process() を、"
                "データ供給を別スレッドにするなら run(producer) を使ってください。"
            )
        self.run()

    def stop(self):
        """ビューワーを閉じる（どのスレッドからでも呼べる）"""
        self._stop_event.set()

        if not self._running:
            return

        if _on_main_thread():
            self._on_close_signal()
        elif self._emitter is not None:
            self._emitter.close_signal.emit()

    def close(self):
        """stop() と同じ"""
        self.stop()

    def is_running(self) -> bool:
        """ビューワーが動いているか"""
        return self._running and not self._stop_event.is_set()

    def __enter__(self) -> HandViewer:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # 姿勢の更新
    # ------------------------------------------------------------------

    def set_hand_pose(self, angles: np.ndarray, target: str = "prediction"):
        """
        手のポーズを設定

        主スレッドから呼んだ場合は、その場で描画まで進める。
        別スレッドから呼んだ場合はキューに積み、GUI側のタイマーが取り出す。

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

        if self._running and _on_main_thread():
            self._drain()
            self._app.processEvents()

    def set_ground_truth(self, angles: np.ndarray):
        """実測値（青い手）を更新"""
        self.set_hand_pose(angles, target="ground_truth")

    def set_prediction(self, angles: np.ndarray):
        """予測値（緑の手）を更新"""
        self.set_hand_pose(angles, target="prediction")

    def set_both(self, ground_truth: np.ndarray, prediction: np.ndarray):
        """両手を同時に更新"""
        self.set_hand_pose(ground_truth, target="ground_truth")
        self.set_hand_pose(prediction, target="prediction")

    def set_angle_scale(self, scale: float):
        """角度スケールを設定"""
        self.angle_scale = scale

    def _put_queue(self, queue: Queue, data: np.ndarray):
        """キューにデータを追加（古いデータは破棄）"""
        while queue.full():
            try:
                queue.get_nowait()
            except Empty:
                break
        try:
            queue.put_nowait(data)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 内部（GUIスレッド）
    # ------------------------------------------------------------------

    def _setup_window(self):
        """ウィンドウをセットアップ"""
        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle(self.title)
        self._window.resize(*self.size)
        self._window.closeEvent = self._on_window_close

        central = QtWidgets.QWidget()
        self._window.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # 3Dビュー
        self._gl_widget = gl.GLViewWidget()
        self._gl_widget.setCameraPosition(distance=6, elevation=25, azimuth=45)
        layout.addWidget(self._gl_widget, stretch=3)

        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        grid.translate(0, 0, -0.5)
        self._gl_widget.addItem(grid)

        axis = gl.GLAxisItem()
        axis.setSize(1, 1, 1)
        self._gl_widget.addItem(axis)

        # 手モデル
        self._hand_model = DualHandModel3D(self._gl_widget)
        self._hand_model.ground_truth.set_visible(self.show_ground_truth)
        self._hand_model.prediction.set_visible(self.show_prediction)

        control = self._create_control_panel()
        layout.addWidget(control, stretch=1)

        self._window.show()

    def _create_control_panel(self) -> QtWidgets.QWidget:
        """コントロールパネル作成"""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

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

        # 凡例（左右は視点で変わるので書かない）
        legend_group = QtWidgets.QGroupBox("Legend")
        legend_layout = QtWidgets.QVBoxLayout(legend_group)
        legend_layout.addWidget(QtWidgets.QLabel("Blue: Ground Truth"))
        legend_layout.addWidget(QtWidgets.QLabel("Green: Prediction"))
        layout.addWidget(legend_group)

        layout.addStretch()

        return panel

    def _drain(self):
        """キューに溜まった姿勢を手モデルに反映する"""
        if not self._running:
            return

        if self._stop_event.is_set():
            self._on_close_signal()
            return

        for queue, hand in (
            (self._gt_queue, self._hand_model.ground_truth),
            (self._pred_queue, self._hand_model.prediction),
        ):
            angles = None
            while True:
                try:
                    angles = queue.get_nowait()
                except Empty:
                    break
            if angles is not None:
                hand.update_from_angles(angles, self.angle_scale)

    def _on_close_signal(self):
        """クローズ要求を受けたとき"""
        if self._timer is not None:
            self._timer.stop()
        if self._window is not None:
            self._window.close()
        self._running = False
        if self._app is not None:
            self._app.quit()

    def _on_window_close(self, event):
        """ウィンドウが閉じられたとき"""
        self._stop_event.set()
        self._running = False
        if self._timer is not None:
            self._timer.stop()
        event.accept()
        if self._app is not None:
            self._app.quit()


class HandViewerContext:
    """
    with で使えるHandViewer

    使用例:
        with HandViewerContext() as viewer:
            for data in stream:
                viewer.set_prediction(data)   # ここで描画も進む
                if not viewer.is_running():
                    break
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.viewer: Optional[HandViewer] = None

    def __enter__(self) -> HandViewer:
        self.viewer = HandViewer(**self.kwargs)
        self.viewer.open()
        return self.viewer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.viewer:
            self.viewer.close()
        return False
