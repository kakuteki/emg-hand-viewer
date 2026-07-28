"""データ読み込みとファイル再生の試験"""

import numpy as np
import pytest

from emg_realtime_viz.core.data_loader import NinaproLoader
from emg_realtime_viz.devices.file_source import NinaproDataSource


def make_segment(subject_id: int, movement: int, n_samples: int = 5) -> dict:
    return {
        "subject_id": subject_id,
        "exercise_id": 1,
        "movement": movement,
        "repetition": 1,
        "emg": np.arange(n_samples * 16, dtype=np.float32).reshape(n_samples, 16),
        "glove": np.zeros((n_samples, 22), dtype=np.float32),
        "n_samples": n_samples,
    }


@pytest.fixture
def dataset(tmp_path):
    segments = [
        make_segment(1, 1),
        make_segment(1, 2),
        make_segment(2, 1),
    ]
    path = tmp_path / "dataset.npz"
    np.savez(path, segments=np.array(segments, dtype=object))
    return path


def test_読み込みと絞り込み(dataset):
    loader = NinaproLoader(str(dataset))

    assert len(loader) == 3
    assert loader.subjects == [1, 2]
    assert loader.movements == [1, 2]
    assert loader.exercises == [1]

    assert len(loader.get_by_subject(1)) == 2
    assert len(loader.get_by_movement(1)) == 2

    segment = loader[0]
    assert segment.n_channels == 16
    assert segment.n_joints == 22


def test_無いファイルは例外(tmp_path):
    with pytest.raises(FileNotFoundError):
        NinaproLoader(str(tmp_path / "missing.npz"))


def test_ファイル再生は指定した条件だけ流す(dataset):
    source = NinaproDataSource(filepath=str(dataset), subject_ids=[1], movements=[1], loop=False)
    assert source.connect()
    assert source.total_segments == 1

    frames = list(source.stream())
    assert len(frames) == 5

    emg, glove, metadata = frames[0]
    assert emg.shape == (1, 16)
    assert glove.shape == (1, 22)
    assert metadata["subject_id"] == 1
    assert metadata["movement"] == 1


def test_条件に合うデータが無ければ接続に失敗(dataset):
    source = NinaproDataSource(filepath=str(dataset), subject_ids=[99])
    assert not source.connect()


def test_ループ再生は先頭に戻る(dataset):
    source = NinaproDataSource(filepath=str(dataset), subject_ids=[1], movements=[1], loop=True)
    source.connect()

    stream = source.stream()
    indices = [next(stream)[2]["sample_idx"] for _ in range(7)]

    assert indices == [0, 1, 2, 3, 4, 0, 1]


def test_リセットで先頭に戻る(dataset):
    """再生中にResetを押しても、次のサンプルが先頭に戻ること"""
    source = NinaproDataSource(filepath=str(dataset), subject_ids=[1], movements=[1])
    source.connect()

    stream = source.stream()
    next(stream)
    next(stream)
    source.reset()

    assert source.current_progress[2] == 0
    assert next(stream)[2]["sample_idx"] == 0
