"""グローブ値の割り当てと正規化の試験"""

import numpy as np
import pytest

from emg_realtime_viz.core.glove import (
    FINGERS,
    FLEXION_COLUMNS,
    FLEXION_COLUMNS_GROUPED,
    GLOVE_MAX,
    GLOVE_MIN,
    N_GLOVE_SENSORS,
    GloveNormalizer,
    glove_to_angles,
    normalize_glove,
    to_finger_angles,
    to_hand_angles,
)


def test_センサの割り当ては指ごとに3個ずつ():
    """CyberGlove IIの屈曲センサは指1本あたり3個しかない"""
    for layout in (FLEXION_COLUMNS, FLEXION_COLUMNS_GROUPED):
        assert set(layout) == set(FINGERS)
        columns = [c for finger in FINGERS for c in layout[finger]]
        assert len(columns) == 15
        assert len(set(columns)) == 15, "列が重複している"
        assert all(0 <= c < N_GLOVE_SENSORS for c in columns)


def test_外転センサを指の曲げに使わない():
    """本体の並びでは3,7,11,15が外転、19が手のひら、20-21が手首"""
    used = {c for finger in FINGERS for c in FLEXION_COLUMNS[finger]}
    for column in (3, 7, 11, 15, 19, 20, 21):
        assert column not in used


def test_割り当てた列だけが角度になる():
    normalized = np.zeros(N_GLOVE_SENSORS)
    normalized[FLEXION_COLUMNS["middle"][1]] = 1.0  # 中指のPIP

    angles = to_finger_angles(normalized)

    assert angles[9] == 1.0  # 中指の2番目
    assert angles.sum() == 1.0, "他の関節まで動いている"


def test_指先の枠は空のまま():
    """屈曲センサが3個なので20次元の4番目は使わない"""
    angles = to_finger_angles(np.ones(N_GLOVE_SENSORS))
    for finger_idx in range(5):
        assert angles[finger_idx * 4 + 3] == 0.0


def test_並びを差し替えられる():
    normalized = np.zeros(N_GLOVE_SENSORS)
    normalized[3] = 1.0  # 本体の並びでは外転、まとめた並びでは人差し指MP

    assert to_finger_angles(normalized).sum() == 0.0
    assert to_finger_angles(normalized, FLEXION_COLUMNS_GROUPED)[4] == 1.0


def test_固定の範囲で下限は0上限は1():
    assert normalize_glove(GLOVE_MIN) == pytest.approx(np.zeros(N_GLOVE_SENSORS))
    assert normalize_glove(GLOVE_MAX) == pytest.approx(np.ones(N_GLOVE_SENSORS))


def test_範囲外は0と1に丸められる():
    assert normalize_glove(GLOVE_MIN - 1000) == pytest.approx(np.zeros(N_GLOVE_SENSORS))
    assert normalize_glove(GLOVE_MAX + 1000) == pytest.approx(np.ones(N_GLOVE_SENSORS))


def test_NaNは0にする():
    assert normalize_glove(np.full(22, np.nan)) == pytest.approx(np.zeros(N_GLOVE_SENSORS))
    assert to_hand_angles(np.full(20, np.nan)) == pytest.approx(np.zeros(20))


def test_センサ数が足りなくても落ちない():
    angles = glove_to_angles(np.zeros(5))
    assert angles.shape == (20,)
    assert np.all((angles >= 0) & (angles <= 1))


def test_モデル出力の受け口():
    # 22次元はグローブと同じ並びとして屈曲センサだけ取り出す
    values = np.zeros(22)
    values[FLEXION_COLUMNS["pinky"][0]] = 1.0
    assert to_hand_angles(values)[16] == 1.0

    # 20次元は手モデルの並びとしてそのまま
    assert to_hand_angles(np.linspace(0, 1, 20)).shape == (20,)

    # 短い場合は0で埋める
    padded = to_hand_angles(np.ones(4))
    assert padded[:4] == pytest.approx(np.ones(4))
    assert padded[4:] == pytest.approx(np.zeros(16))

    # 0-1の外は丸める
    assert to_hand_angles(np.full(20, 5.0)) == pytest.approx(np.ones(20))
    assert to_hand_angles(np.full(20, -5.0)) == pytest.approx(np.zeros(20))


class TestGloveNormalizer:
    def make_rows(self):
        rows = np.zeros((100, N_GLOVE_SENSORS))
        for column in range(N_GLOVE_SENSORS):
            rows[:, column] = np.linspace(100 * column, 100 * column + 500, 100)
        return rows

    def test_あてはめる前は固定の範囲(self):
        normalizer = GloveNormalizer()
        assert not normalizer.is_fitted
        assert normalizer(GLOVE_MIN) == pytest.approx(np.zeros(20))

    def test_データから範囲を決める(self):
        rows = self.make_rows()
        normalizer = GloveNormalizer(low_percentile=0, high_percentile=100).fit(rows)

        assert normalizer.is_fitted
        assert normalizer(rows[0]) == pytest.approx(np.zeros(20))
        assert normalizer(rows[-1])[:3] == pytest.approx(np.ones(3))

    def test_動かないセンサは0のまま(self):
        rows = np.full((50, N_GLOVE_SENSORS), 7.0)
        normalizer = GloveNormalizer().fit(rows)

        assert normalizer(rows[0]) == pytest.approx(np.zeros(20))

    def test_外れ値に引きずられない(self):
        rows = self.make_rows()
        rows[0, :] = 1e9  # 1サンプルだけ極端な値

        normalizer = GloveNormalizer().fit(rows)
        angles = normalizer(rows[50])

        assert np.any(angles > 0.1), "外れ値のせいで全部0に潰れている"

    def test_空のデータでは何もしない(self):
        normalizer = GloveNormalizer().fit(np.zeros((0, N_GLOVE_SENSORS)))
        assert not normalizer.is_fitted

    def test_センサ数が少ないデータ(self):
        normalizer = GloveNormalizer().fit(np.random.rand(30, 10))
        assert normalizer(np.zeros(10)).shape == (20,)
