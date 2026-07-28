"""HandViewerの約束ごとの試験（ウィンドウは開かない）"""

import threading

import numpy as np
import pytest

pytest.importorskip("PyQt5", reason="PyQt5が無い環境ではHandViewerを組み立てられない")
pytest.importorskip("pyqtgraph", reason="pyqtgraphが必要")

from emg_realtime_viz import HandViewer  # noqa: E402


def test_別スレッドでGUIを回す指定は例外():
    """Qtの制約で動かないため、黙って壊れるのではなく理由を返す"""
    viewer = HandViewer()
    with pytest.raises(RuntimeError) as e:
        viewer.start(blocking=False)
    assert "run(producer)" in str(e.value)


def test_主スレッド以外からopenすると例外():
    viewer = HandViewer()
    errors = []

    def worker():
        try:
            viewer.open()
        except RuntimeError as e:
            errors.append(str(e))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)

    assert errors, "別スレッドからのopen()が通ってしまった"
    assert "主スレッド" in errors[0]


def test_開く前でも姿勢を積める():
    """ウィンドウが無い間の更新は捨てずにキューへ入れる"""
    viewer = HandViewer()
    viewer.set_prediction(np.zeros(20))
    viewer.set_ground_truth(np.ones(20))

    assert viewer._pred_queue.qsize() == 1
    assert viewer._gt_queue.qsize() == 1


def test_キューは古いものから捨てる():
    viewer = HandViewer()
    for i in range(50):
        viewer.set_prediction(np.full(20, i / 50))

    assert viewer._pred_queue.qsize() <= 10


def test_開いていなければ動いていない():
    viewer = HandViewer()
    assert not viewer.is_running()
    assert not viewer.process()
