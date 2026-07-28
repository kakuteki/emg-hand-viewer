"""特徴量抽出の試験"""

import numpy as np
import pytest

from emg_realtime_viz.core.feature_extractor import FeatureExtractor, SlidingWindowExtractor


def make_extractor(**kwargs) -> FeatureExtractor:
    params = {"window_size": 4, "n_channels": 3, "features": ["mav", "rms"]}
    params.update(kwargs)
    return FeatureExtractor(**params)


def fill(extractor: FeatureExtractor, value: float = 1.0):
    for _ in range(extractor.window_size):
        extractor.update(np.full(extractor.n_channels, value))


def test_窓が埋まるまでは準備できていない():
    extractor = make_extractor()
    assert not extractor.is_ready

    extractor.update(np.ones(3))
    assert not extractor.is_ready
    assert extractor.buffer_fill == pytest.approx(0.25)

    fill(extractor)
    assert extractor.is_ready
    assert extractor.buffer_fill == pytest.approx(1.0)


def test_準備前に取り出すと例外():
    extractor = make_extractor()
    with pytest.raises(RuntimeError):
        extractor.extract()


def test_チャンネル数が違う入力は例外():
    extractor = make_extractor()
    with pytest.raises(ValueError):
        extractor.update(np.ones(5))


def test_特徴量の次元は特徴量数かけるチャンネル数():
    extractor = make_extractor()
    fill(extractor)

    assert extractor.feature_dim == 6
    assert extractor.extract().shape == (6,)
    assert len(extractor.feature_names) == 6


def test_既知の入力で値が合う():
    extractor = make_extractor(features=["mav", "rms", "iemg", "wl"])
    for value in (-1.0, 1.0, -1.0, 1.0):
        extractor.update(np.full(3, value))

    result = extractor.extract_dict()

    assert result["mav"] == pytest.approx(np.ones(3))
    assert result["rms"] == pytest.approx(np.ones(3))
    assert result["iemg"] == pytest.approx(np.full(3, 4.0))
    # 2ずつ3回変化する
    assert result["wl"] == pytest.approx(np.full(3, 6.0))


def test_ゼロ交差はノイズを数えない():
    extractor = make_extractor(window_size=6, n_channels=1, features=["zc"])
    # 平均のまわりを微小に揺れるだけ（閾値0.01より小さい）
    for value in (0.0, 0.001, -0.001, 0.001, -0.001, 0.0):
        extractor.update(np.array([value]))
    assert extractor.extract()[0] == 0.0

    # はっきり振れれば数える
    extractor.reset()
    for value in (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0):
        extractor.update(np.array([value]))
    assert extractor.extract()[0] == 5.0


def test_傾きの符号変化もノイズを数えない():
    extractor = make_extractor(window_size=6, n_channels=1, features=["ssc"])
    for value in (0.0, 0.001, 0.0, 0.001, 0.0, 0.001):
        extractor.update(np.array([value]))
    assert extractor.extract()[0] == 0.0

    extractor.reset()
    for value in (0.0, 1.0, 0.0, 1.0, 0.0, 1.0):
        extractor.update(np.array([value]))
    assert extractor.extract()[0] == 4.0


def test_知らない特徴量名は例外():
    extractor = make_extractor(features=["mav", "nonexistent"])
    fill(extractor)
    with pytest.raises(ValueError):
        extractor.extract()


def test_リセットで空になる():
    extractor = make_extractor()
    fill(extractor)
    extractor.reset()
    assert not extractor.is_ready


def test_全特徴量が計算できる():
    extractor = make_extractor(window_size=8, features=list(FeatureExtractor.AVAILABLE_FEATURES))
    rng = np.random.default_rng(0)
    for _ in range(8):
        extractor.update(rng.normal(size=3))

    values = extractor.extract()
    assert values.shape == (len(FeatureExtractor.AVAILABLE_FEATURES) * 3,)
    assert np.all(np.isfinite(values))


def test_スライディング窓はhopごとに返す():
    extractor = SlidingWindowExtractor(window_size=4, hop_size=2, n_channels=3, features=["mav"])

    returned = [extractor.update(np.ones(3)) for _ in range(6)]

    # 窓が埋まるまではNone、その後は2回に1回返る
    assert returned[0] is None
    assert returned[1] is None
    assert returned[2] is None
    assert returned[3] is not None
    assert returned[4] is None
    assert returned[5] is not None
