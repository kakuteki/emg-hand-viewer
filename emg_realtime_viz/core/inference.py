"""
Inference Module
================

学習済みモデルの読み込みと、手モデルへ渡す推論関数の生成

推論関数はどれも「EMGの1フレーム（生値）を受け取り、
20次元の関節角度（0-1）を返す」という同じ形にそろえてある。
入力の形をアプリごとに変えていたために、
学習済みモデルが通らない状態になっていたのを直したもの。
"""

from __future__ import annotations

import contextlib
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .glove import N_HAND_ANGLES, to_hand_angles

# 学習時の窓長（20サンプル = 100ms @ 200Hz）
DEFAULT_WINDOW_SIZE = 20

# PyTorchは重いので、必要になった時点で読み込む
_TORCH = None
_TORCH_CHECKED = False


def _import_torch():
    """PyTorchを遅延読み込みする。入っていなければNone"""
    global _TORCH, _TORCH_CHECKED

    if not _TORCH_CHECKED:
        _TORCH_CHECKED = True
        try:
            import torch

            _TORCH = torch
        except ImportError:
            _TORCH = None

    return _TORCH


class TorchInference:
    """
    PyTorchモデルを1フレーム入力の推論関数として包む

    学習済みモデルの入力の形はモデルによって違うため、
    最初の呼び出しで次の順に試し、通った形を覚えて以後それを使う。

        1. (1, window_size, n_channels)  時系列モデル（LSTM等）
        2. (1, n_channels)               1フレーム入力のモデル
        3. (1, window_size * n_channels) 窓を平らに並べたモデル

    Parameters
    ----------
    model : torch.nn.Module
        評価モードにした学習済みモデル
    device : torch.device
        実行デバイス
    window_size : int
        時系列モデルに渡す窓長
    smoothing : float
        前フレームの重み（0で平滑化なし、1で更新なし）
    """

    INPUT_MODES = ("window", "frame", "flat")

    def __init__(
        self, model, device, window_size: int = DEFAULT_WINDOW_SIZE, smoothing: float = 0.7
    ):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.smoothing = smoothing

        self._buffer: deque = deque(maxlen=window_size)
        self._input_mode: Optional[str] = None
        self._prev: Optional[np.ndarray] = None
        self._last_error: Optional[str] = None

    @property
    def input_mode(self) -> Optional[str]:
        """実際に通った入力の形（未確定ならNone）"""
        return self._input_mode

    @property
    def last_error(self) -> Optional[str]:
        """直近の失敗理由"""
        return self._last_error

    def reset(self):
        """内部の窓と平滑化の履歴を捨てる"""
        self._buffer.clear()
        self._prev = None

    def _prepare_array(self, mode: str) -> np.ndarray:
        """指定した形の入力配列を作る"""
        window = np.array(self._buffer, dtype=np.float32)  # (T, ch)

        # 窓が埋まるまでは先頭のフレームで前を埋める
        if len(window) < self.window_size:
            pad = np.repeat(window[:1], self.window_size - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)

        if mode == "window":
            return window[np.newaxis, :, :]
        if mode == "frame":
            return window[-1][np.newaxis, :]
        if mode == "flat":
            return window.reshape(1, -1)
        raise ValueError(f"Unknown input mode: {mode}")

    def _to_tensor(self, array: np.ndarray):
        """PyTorchがあればテンソルに、無ければそのまま渡す"""
        torch = _import_torch()
        if torch is None:
            return array
        return torch.tensor(array, dtype=torch.float32).to(self.device)

    def _no_grad(self):
        torch = _import_torch()
        return torch.no_grad() if torch is not None else contextlib.nullcontext()

    def _forward(self, mode: str) -> np.ndarray:
        with self._no_grad():
            output = self.model(self._to_tensor(self._prepare_array(mode)))

        if isinstance(output, (tuple, list)):
            output = output[0]

        for method in ("detach", "cpu", "numpy"):
            if hasattr(output, method):
                output = getattr(output, method)()

        return np.asarray(output)

    def __call__(self, emg_frame: np.ndarray) -> np.ndarray:
        """
        EMGの1フレームから関節角度を推論する

        Parameters
        ----------
        emg_frame : np.ndarray
            EMGの生値 (n_channels,) または (1, n_channels)

        Returns
        -------
        np.ndarray
            関節角度 (20,)、各値は0-1

        Raises
        ------
        RuntimeError
            どの入力の形でもモデルが動かなかった場合
        """
        frame = np.asarray(emg_frame, dtype=np.float32).flatten()
        self._buffer.append(frame)

        modes = (self._input_mode,) if self._input_mode else self.INPUT_MODES
        raw = None
        errors = []

        for mode in modes:
            try:
                raw = self._forward(mode)
                self._input_mode = mode
                break
            except Exception as e:  # 入力の形が合わないときだけ次を試す
                errors.append(f"{mode}: {e}")

        if raw is None:
            self._last_error = " / ".join(errors)
            raise RuntimeError(f"モデルにどの入力の形も通りませんでした（{self._last_error}）")

        angles = to_hand_angles(raw)

        if self._prev is not None:
            angles = (1.0 - self.smoothing) * angles + self.smoothing * self._prev
        self._prev = angles

        return angles.astype(np.float32)


class SimpleRegressor:
    """state_dictだけが保存されていた場合に組み立てる素朴な全結合モデル"""

    def __new__(cls, input_dim: int, output_dim: int):
        from torch import nn

        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
        )


def _torch_load(path: Path, device):
    """
    PyTorch 2.6以降は torch.load が既定で重みしか読まないため、
    モデル本体が入ったファイルは weights_only=False を明示しないと開けない。

    Notes
    -----
    weights_only=False は保存ファイル内のコードを実行しうる。
    自分で作ったモデルなど、出所の分かるファイルにだけ使うこと。
    """
    torch = _import_torch()

    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        # weights_only 引数を持たない古いPyTorch
        return torch.load(str(path), map_location=device)


def load_torch_model(
    model_path: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    smoothing: float = 0.7,
) -> TorchInference:
    """
    学習済みモデル（.pth）を読み込んで推論関数を作る

    モデル本体が保存されていればそのまま使い、
    state_dictだけならSimpleRegressorに読み込ませる。

    Parameters
    ----------
    model_path : str
        .pth ファイルのパス
    window_size : int
        時系列モデルに渡す窓長
    smoothing : float
        前フレームの重み

    Returns
    -------
    TorchInference

    Raises
    ------
    ImportError
        PyTorchが入っていない場合
    RuntimeError
        ファイルを読めなかった場合
    """
    torch = _import_torch()
    if torch is None:
        raise ImportError("PyTorchが必要です: pip install torch")

    path = Path(model_path)
    if not path.exists():
        raise RuntimeError(f"モデルファイルが見つかりません: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = _torch_load(path, device)

    if isinstance(obj, dict):
        state_dict = obj.get("state_dict", obj)
        keys = list(state_dict.keys())
        if not keys:
            raise RuntimeError("state_dictが空です")

        first_weight = state_dict[keys[0]]
        if first_weight.ndim < 2:
            raise RuntimeError("state_dictから入力次元を推定できません")

        input_dim = first_weight.shape[1]
        model = SimpleRegressor(input_dim, 22)
        model.load_state_dict(state_dict)
        model = model.to(device)
    else:
        model = obj.to(device) if hasattr(obj, "to") else obj

    model.eval()
    return TorchInference(model, device, window_size=window_size, smoothing=smoothing)


def energy_demo_model(smoothing: float = 0.7) -> Callable[[np.ndarray], np.ndarray]:
    """
    学習済みモデルが無いときのデモ用推論

    EMGの大きさをそのまま指の曲げに割り当てるだけのもの。
    チャンネルを3つずつ5本の指に配り、根元より先端をわずかに強く曲げる。
    """
    state = {"prev": np.zeros(N_HAND_ANGLES, dtype=np.float32)}

    def inference(emg_frame: np.ndarray) -> np.ndarray:
        energy = np.abs(np.asarray(emg_frame, dtype=np.float32).flatten())
        energy = np.clip(energy / 100.0, 0.0, 1.0)

        angles = np.zeros(N_HAND_ANGLES, dtype=np.float32)
        for finger in range(5):
            ch_start = finger * 3
            ch_end = min(ch_start + 3, len(energy))
            finger_energy = (
                float(np.mean(energy[ch_start:ch_end])) if ch_start < len(energy) else 0.0
            )

            for joint in range(4):
                angles[finger * 4 + joint] = finger_energy * (0.7 + joint * 0.1)

        smoothed = (1.0 - smoothing) * angles + smoothing * state["prev"]
        state["prev"] = smoothed
        return np.clip(smoothed, 0.0, 1.0)

    return inference


def wave_demo_model() -> Callable[[np.ndarray], np.ndarray]:
    """
    表示の確認用に、指を順番に波打たせるだけのデモ推論

    EMGの大きさは曲げ量に少しだけ効く。推論の中身は時間の関数。
    """
    start_time = time.time()

    def inference(emg_frame: np.ndarray) -> np.ndarray:
        energy = float(np.mean(np.abs(np.asarray(emg_frame, dtype=np.float32))))
        energy = min(energy, 2.0)

        t = time.time() - start_time
        angles = np.zeros(N_HAND_ANGLES, dtype=np.float32)

        for finger in range(5):
            phase = finger * 0.3
            for joint in range(4):
                delay = joint * 0.15
                wave = np.sin(t * 2 + phase + delay)
                angles[finger * 4 + joint] = 0.3 + 0.3 * (wave * 0.5 + 0.5) + 0.2 * energy

        angles += np.random.randn(N_HAND_ANGLES).astype(np.float32) * 0.02
        return np.clip(angles, 0.0, 1.0)

    return inference
