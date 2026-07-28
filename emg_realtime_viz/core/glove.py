"""
Glove Data Module
=================

Ninapro DB5のデータグローブ（22センサ）の値を、
手モデルが受け取る20次元の関節角度（0-1）に変換する

このモジュールがセンサの割り当てと正規化の唯一の定義元。
以前は可視化側とアプリ側に別々の表が書かれており、
同じデータでも手の形が変わる原因になっていた。

センサの構成
------------
DB5が使う CyberGlove II の22センサは、指1本あたり屈曲3個（計15個）、
指の間の外転4個、手のひらのそり1個、手首2個で構成される。
「指1本あたり4個 x 5本 + 手首2個」ではない。

    3 x 5 + 4 + 1 + 2 = 22

したがって、先頭20列をそのまま指の関節と見なすと、外転センサや
手のひらのそりを指の曲げとして扱ってしまう。

列の並び
--------
CyberGloveの取扱説明書に沿った並び（親指から小指へ、各指の間に
外転センサが入る）を既定にしている。Ninapro公式の説明ページ
（ninapro.hevs.ch/node/123）は現在404で、並びを断定できる一次資料が
確認できていないため、屈曲センサをまとめて並べる実装向けに
FLEXION_COLUMNS_GROUPED も用意してある。

実データで確かめる場合は、指を1本ずつ曲げた区間で各列との相関を見て、
どの列がどの指に対応するかを確認すること。

値の大きさ
----------
DB5のグローブ値は未校正で、角度に比例する生の値である。度数を仮定した
固定の範囲は当てにならないので、既定ではデータから範囲を求める
GloveNormalizer を使う。固定の範囲は最後の手段。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

N_GLOVE_SENSORS = 22
N_HAND_ANGLES = 20

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# CyberGlove本体の並び（親指ひねり, 親指MP, 親指IP, 親指-人差し指の外転, 人差し指MP, ...）
FLEXION_COLUMNS: Dict[str, Sequence[int]] = {
    "thumb": (0, 1, 2),  # ひねり(CMC), MP, IP
    "index": (4, 5, 6),  # MP, PIP, DIP
    "middle": (8, 9, 10),
    "ring": (12, 13, 14),
    "pinky": (16, 17, 18),
}

# 屈曲センサ15個を先にまとめて並べる実装向け
FLEXION_COLUMNS_GROUPED: Dict[str, Sequence[int]] = {
    "thumb": (0, 1, 2),
    "index": (3, 4, 5),
    "middle": (6, 7, 8),
    "ring": (9, 10, 11),
    "pinky": (12, 13, 14),
}


def _build_range(thumb, finger, wrist) -> np.ndarray:
    """22センサぶんの範囲を組み立てる（最後の手段の固定値用）"""
    return np.array(list(thumb) + list(finger) * 4 + list(wrist), dtype=np.float64)


# データから範囲を求められないときだけ使う固定の範囲。
# 公式仕様ではなく、実データの分布から決めた目安。
GLOVE_MIN = _build_range(
    thumb=(-30.0, -30.0, -10.0, -10.0),
    finger=(-20.0, -10.0, -10.0, -10.0),
    wrist=(0.0, 0.0),
)

GLOVE_MAX = _build_range(
    thumb=(100.0, 100.0, 100.0, 100.0),
    finger=(120.0, 100.0, 100.0, 100.0),
    wrist=(50.0, 50.0),
)


def _as_sensor_row(glove: np.ndarray) -> np.ndarray:
    """22要素にそろえる（足りなければ0埋め、多ければ切り捨て）"""
    values = np.asarray(glove, dtype=np.float64).flatten()

    if len(values) < N_GLOVE_SENSORS:
        padded = np.zeros(N_GLOVE_SENSORS)
        padded[: len(values)] = values
        return padded

    return values[:N_GLOVE_SENSORS]


def normalize_glove(
    glove: np.ndarray,
    minimum: Optional[np.ndarray] = None,
    maximum: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    データグローブの生値を0-1に正規化する

    Parameters
    ----------
    glove : np.ndarray
        グローブの生値。22要素を想定
    minimum, maximum : np.ndarray, optional
        センサごとの範囲。省略すると固定の目安を使う

    Returns
    -------
    np.ndarray
        0-1に収めた値 (22,)
    """
    values = _as_sensor_row(glove)

    low = GLOVE_MIN if minimum is None else np.asarray(minimum, dtype=np.float64)
    high = GLOVE_MAX if maximum is None else np.asarray(maximum, dtype=np.float64)

    span = high - low
    span[span == 0] = 1.0  # 動かないセンサは0扱いにする

    normalized = (values - low) / span
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(normalized, 0.0, 1.0)


def to_finger_angles(
    normalized: np.ndarray, layout: Optional[Dict[str, Sequence[int]]] = None
) -> np.ndarray:
    """
    0-1に正規化済みのセンサ値を、手モデル用の20次元に並べ替える

    指1本につき屈曲センサは3個しかないので、20次元のうち各指の4番目
    （指先側）は0のままにする。手モデルの順運動学も4本指については
    3個しか使わないため、実質の欠落は親指の先だけになる。

    Parameters
    ----------
    normalized : np.ndarray
        0-1のセンサ値 (22,)
    layout : dict, optional
        指ごとの列番号。省略すると FLEXION_COLUMNS

    Returns
    -------
    np.ndarray
        関節角度 (20,)
    """
    columns = FLEXION_COLUMNS if layout is None else layout
    values = np.asarray(normalized, dtype=np.float64).flatten()

    angles = np.zeros(N_HAND_ANGLES, dtype=np.float32)
    for finger_idx, finger in enumerate(FINGERS):
        base = finger_idx * 4
        for joint_idx, column in enumerate(columns[finger]):
            if column < len(values):
                angles[base + joint_idx] = values[column]

    return angles


def glove_to_angles(
    glove: np.ndarray,
    minimum: Optional[np.ndarray] = None,
    maximum: Optional[np.ndarray] = None,
    layout: Optional[Dict[str, Sequence[int]]] = None,
) -> np.ndarray:
    """
    グローブの生値から手モデル用の20次元角度を作る

    Parameters
    ----------
    glove : np.ndarray
        グローブの生値 (22,)
    minimum, maximum : np.ndarray, optional
        センサごとの範囲
    layout : dict, optional
        指ごとの列番号

    Returns
    -------
    np.ndarray
        関節角度 (20,)、各値は0-1
    """
    return to_finger_angles(normalize_glove(glove, minimum, maximum), layout)


def to_hand_angles(values: np.ndarray) -> np.ndarray:
    """
    すでに0-1になっている推論結果を20次元の角度にそろえる

    22次元（グローブと同じ並び）で来た場合は屈曲センサだけを取り出す。
    20次元なら手モデルの並びとみなしてそのまま使う。

    Parameters
    ----------
    values : np.ndarray
        モデル出力

    Returns
    -------
    np.ndarray
        関節角度 (20,)、各値は0-1
    """
    array = np.asarray(values, dtype=np.float32).flatten()
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array, 0.0, 1.0)

    if len(array) == N_GLOVE_SENSORS:
        return to_finger_angles(array)

    if len(array) >= N_HAND_ANGLES:
        return array[:N_HAND_ANGLES]

    angles = np.zeros(N_HAND_ANGLES, dtype=np.float32)
    angles[: len(array)] = array
    return angles


class GloveNormalizer:
    """
    読み込んだデータからセンサごとの範囲を求めて正規化する

    DB5のグローブ値は未校正で、単位も基準もデータ次第なので、
    度数を仮定した固定の範囲より実データの分布のほうが当てになる。

    Parameters
    ----------
    low_percentile, high_percentile : float
        範囲として使う分位点。外れ値に引きずられないよう端を落とす
    layout : dict, optional
        指ごとの列番号
    """

    def __init__(
        self,
        low_percentile: float = 2.0,
        high_percentile: float = 98.0,
        layout: Optional[Dict[str, Sequence[int]]] = None,
    ):
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.layout = layout

        self.minimum: Optional[np.ndarray] = None
        self.maximum: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self.minimum is not None

    def fit(self, rows: np.ndarray) -> GloveNormalizer:
        """
        センサごとの範囲を求める

        Parameters
        ----------
        rows : np.ndarray
            グローブ値 (n_samples, n_sensors)

        Returns
        -------
        GloveNormalizer
            自分自身
        """
        data = np.asarray(rows, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] == 0:
            return self

        if data.shape[1] < N_GLOVE_SENSORS:
            padded = np.zeros((data.shape[0], N_GLOVE_SENSORS))
            padded[:, : data.shape[1]] = data
            data = padded
        else:
            data = data[:, :N_GLOVE_SENSORS]

        data = np.nan_to_num(data, nan=0.0)

        low = np.percentile(data, self.low_percentile, axis=0)
        high = np.percentile(data, self.high_percentile, axis=0)

        # 動かないセンサは常に0にする（span=0のときの割り算を避ける）
        flat = high <= low
        high = np.where(flat, low + 1.0, high)

        self.minimum = low
        self.maximum = high
        return self

    def __call__(self, glove: np.ndarray) -> np.ndarray:
        """1フレームぶんを20次元の角度にする"""
        return glove_to_angles(glove, self.minimum, self.maximum, self.layout)
