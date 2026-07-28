"""データソース共通の約束ごとの試験"""

import numpy as np
import pytest

from emg_realtime_viz.devices.base import DataSource
from emg_realtime_viz.devices.file_source import FilePlaybackSource
from emg_realtime_viz.devices.myo_source import SimulatedMyoSource


class DummySource(DataSource):
    def stream(self):
        yield np.zeros((1, self.n_channels)), None, {}


@pytest.mark.parametrize(
    "source",
    [
        DummySource(),
        SimulatedMyoSource(n_channels=8),
        FilePlaybackSource(emg_data=np.zeros((4, 8), dtype=np.float32)),
    ],
)
def test_どのデータソースもresetを持つ(source):
    """画面のResetボタンは種類を問わずreset()を呼ぶ"""
    source.reset()


def test_接続状態の出入り():
    source = DummySource()
    assert not source.is_connected

    with source as opened:
        assert opened.is_connected

    assert not source.is_connected


def test_ファイル再生はループする():
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    source = FilePlaybackSource(emg_data=data, loop=True)
    source.connect()

    stream = source.stream()
    seen = [float(next(stream)[0][0, 0]) for _ in range(5)]

    assert seen == [0.0, 2.0, 4.0, 0.0, 2.0]


def test_ファイル再生はリセットで先頭に戻る():
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    source = FilePlaybackSource(emg_data=data, loop=False)
    source.connect()

    stream = source.stream()
    next(stream)
    next(stream)
    source.reset()

    assert float(next(stream)[0][0, 0]) == 0.0
