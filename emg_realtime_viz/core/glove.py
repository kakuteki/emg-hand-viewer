"""
Glove Data Module
=================

Ninapro DB5のデータグローブ（22センサ）の値を、
手モデルが受け取る20次元の関節角度（0-1）に変換する

このモジュールが正規化範囲の唯一の定義元。
以前は可視化側とアプリ側に別々の表が書かれており、
同じデータでも手の形が変わる原因になっていた。
"""

from __future__ import annotations

import numpy as np


def _build_range(thumb, finger, wrist) -> np.ndarray:
    """
    22センサぶんの範囲を組み立てる

    人差し指から小指までは同じ範囲を使うので4回並べる。

    Parameters
    ----------
    thumb : tuple
        親指の4関節ぶん
    finger : tuple
        指1本の4関節ぶん（人差し指から小指まで共通）
    wrist : tuple
        手首の2センサぶん
    """
    return np.array(list(thumb) + list(finger) * 4 + list(wrist), dtype=np.float64)


# データグローブの生値の範囲（度）
# Ninapro DB5 は CyberGlove II（22センサ）を使用。
# 下の値は実データの分布から決めた目安であり、公式仕様ではない。
GLOVE_MIN = _build_range(
    thumb=(-30.0, -30.0, -10.0, -10.0),
    finger=(-20.0, -10.0, -10.0, -10.0),
    wrist=(0.0, 0.0),  # 手首は手モデルでは使わない
)

GLOVE_MAX = _build_range(
    thumb=(100.0, 100.0, 100.0, 100.0),
    finger=(120.0, 100.0, 100.0, 100.0),
    wrist=(50.0, 50.0),
)

N_GLOVE_SENSORS = len(GLOVE_MIN)
N_HAND_ANGLES = 20


def normalize_glove(glove: np.ndarray) -> np.ndarray:
    """
    データグローブの生値を0-1に正規化する

    Parameters
    ----------
    glove : np.ndarray
        グローブの生値。22要素を想定（足りない分は0埋め）

    Returns
    -------
    np.ndarray
        0-1に収めた値 (22,)
    """
    values = np.asarray(glove, dtype=np.float64).flatten()

    if len(values) < N_GLOVE_SENSORS:
        padded = np.zeros(N_GLOVE_SENSORS)
        padded[: len(values)] = values
        values = padded
    else:
        values = values[:N_GLOVE_SENSORS]

    normalized = (values - GLOVE_MIN) / (GLOVE_MAX - GLOVE_MIN)
    return np.clip(normalized, 0.0, 1.0)


def glove_to_angles(glove: np.ndarray) -> np.ndarray:
    """
    グローブの生値から手モデル用の20次元角度を作る

    先頭20要素（指5本 x 4関節）を使い、手首の2センサは捨てる。

    Parameters
    ----------
    glove : np.ndarray
        グローブの生値 (22,)

    Returns
    -------
    np.ndarray
        関節角度 (20,)、各値は0-1
    """
    return normalize_glove(glove)[:N_HAND_ANGLES].astype(np.float32)


def to_hand_angles(values: np.ndarray) -> np.ndarray:
    """
    すでに0-1になっている出力を20次元の角度にそろえる

    22次元（グローブと同じ並び）で来た場合は先頭20要素を取り、
    20次元より短い場合は0で埋める。推論モデルの出力を
    手モデルに渡す前の受け口として使う。

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
    array = np.clip(array, 0.0, 1.0)

    if len(array) >= N_HAND_ANGLES:
        return array[:N_HAND_ANGLES]

    angles = np.zeros(N_HAND_ANGLES, dtype=np.float32)
    angles[: len(array)] = array
    return angles
