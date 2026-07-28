"""グローブ値の正規化の試験"""

import numpy as np
import pytest

from emg_realtime_viz.core.glove import (
    GLOVE_MAX,
    GLOVE_MIN,
    N_GLOVE_SENSORS,
    glove_to_angles,
    normalize_glove,
    to_hand_angles,
)


def test_下限は0上限は1になる():
    assert normalize_glove(GLOVE_MIN) == pytest.approx(np.zeros(N_GLOVE_SENSORS))
    assert normalize_glove(GLOVE_MAX) == pytest.approx(np.ones(N_GLOVE_SENSORS))


def test_範囲外は0と1に丸められる():
    values = normalize_glove(GLOVE_MIN - 1000)
    assert values == pytest.approx(np.zeros(N_GLOVE_SENSORS))

    values = normalize_glove(GLOVE_MAX + 1000)
    assert values == pytest.approx(np.ones(N_GLOVE_SENSORS))


def test_手モデル用は20次元():
    angles = glove_to_angles(GLOVE_MAX)
    assert angles.shape == (20,)
    assert angles.dtype == np.float32
    assert np.all(angles == 1.0)


def test_センサ数が足りなくても落ちない():
    angles = glove_to_angles(np.zeros(5))
    assert angles.shape == (20,)
    assert np.all((angles >= 0) & (angles <= 1))


def test_モデル出力の受け口():
    # 22次元は先頭20要素を使う
    assert to_hand_angles(np.linspace(0, 1, 22)).shape == (20,)
    # 短い場合は0で埋める
    padded = to_hand_angles(np.ones(4))
    assert padded.shape == (20,)
    assert padded[:4] == pytest.approx(np.ones(4))
    assert padded[4:] == pytest.approx(np.zeros(16))
    # 0-1の外は丸める
    assert to_hand_angles(np.full(20, 5.0)) == pytest.approx(np.ones(20))
    assert to_hand_angles(np.full(20, -5.0)) == pytest.approx(np.zeros(20))


def test_正規化の表は22要素で下限より上限が大きい():
    assert len(GLOVE_MIN) == len(GLOVE_MAX) == N_GLOVE_SENSORS
    assert np.all(GLOVE_MAX > GLOVE_MIN)
