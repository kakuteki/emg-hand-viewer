"""
Feature Extractor Module
========================

EMG信号からの特徴量抽出
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from collections import deque


class FeatureExtractor:
    """
    リアルタイム特徴量抽出器

    Parameters
    ----------
    window_size : int
        特徴量計算のウィンドウサイズ（サンプル数）
    n_channels : int
        EMGチャンネル数
    features : list of str, optional
        抽出する特徴量のリスト。Noneの場合はデフォルト

    Examples
    --------
    >>> extractor = FeatureExtractor(window_size=20, n_channels=16)
    >>> extractor.update(emg_sample)  # shape: (16,)
    >>> if extractor.is_ready:
    ...     features = extractor.extract()
    """

    AVAILABLE_FEATURES = [
        'mav', 'rms', 'var', 'wl', 'zc', 'ssc',
        'iemg', 'dasdv', 'tkeo', 'mad', 'nle'
    ]

    def __init__(
        self,
        window_size: int = 20,
        n_channels: int = 16,
        features: Optional[List[str]] = None
    ):
        self.window_size = window_size
        self.n_channels = n_channels
        self.features = features or ['mav', 'rms', 'var', 'wl', 'tkeo', 'mad']

        # バッファの初期化
        self._buffer = deque(maxlen=window_size)

        # 特徴量計算関数の登録
        self._feature_funcs: Dict[str, Callable] = {
            'mav': self._compute_mav,
            'rms': self._compute_rms,
            'var': self._compute_var,
            'wl': self._compute_wl,
            'zc': self._compute_zc,
            'ssc': self._compute_ssc,
            'iemg': self._compute_iemg,
            'dasdv': self._compute_dasdv,
            'tkeo': self._compute_tkeo,
            'mad': self._compute_mad,
            'nle': self._compute_nle,
        }

    def reset(self):
        """バッファをリセット"""
        self._buffer.clear()

    def update(self, sample: np.ndarray):
        """
        新しいサンプルをバッファに追加

        Parameters
        ----------
        sample : np.ndarray
            EMGサンプル (n_channels,) または (1, n_channels)
        """
        sample = np.asarray(sample).flatten()
        if len(sample) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {len(sample)}")
        self._buffer.append(sample)

    @property
    def is_ready(self) -> bool:
        """特徴量計算に十分なサンプルがあるか"""
        return len(self._buffer) >= self.window_size

    @property
    def buffer_fill(self) -> float:
        """バッファの充填率 (0.0 - 1.0)"""
        return len(self._buffer) / self.window_size

    def get_window(self) -> np.ndarray:
        """現在のウィンドウデータを取得"""
        return np.array(self._buffer)

    def extract(self) -> np.ndarray:
        """
        特徴量を抽出

        Returns
        -------
        features : np.ndarray
            特徴量ベクトル (n_features * n_channels,)
        """
        if not self.is_ready:
            raise RuntimeError("Buffer not full. Call update() more times.")

        window = self.get_window()  # (window_size, n_channels)
        feature_vectors = []

        for feat_name in self.features:
            if feat_name not in self._feature_funcs:
                raise ValueError(f"Unknown feature: {feat_name}")
            feat_vec = self._feature_funcs[feat_name](window)
            feature_vectors.append(feat_vec)

        return np.concatenate(feature_vectors)

    def extract_dict(self) -> Dict[str, np.ndarray]:
        """
        特徴量を辞書形式で抽出

        Returns
        -------
        features : dict
            特徴量名をキー、値をベクトルとする辞書
        """
        if not self.is_ready:
            raise RuntimeError("Buffer not full.")

        window = self.get_window()
        return {
            feat_name: self._feature_funcs[feat_name](window)
            for feat_name in self.features
        }

    @property
    def feature_dim(self) -> int:
        """特徴量ベクトルの次元数"""
        return len(self.features) * self.n_channels

    @property
    def feature_names(self) -> List[str]:
        """特徴量名のリスト（チャンネル展開済み）"""
        names = []
        for feat in self.features:
            for ch in range(self.n_channels):
                names.append(f"{feat}_ch{ch}")
        return names

    # =================================================================
    # 特徴量計算関数
    # =================================================================

    @staticmethod
    def _compute_mav(window: np.ndarray) -> np.ndarray:
        """Mean Absolute Value"""
        return np.mean(np.abs(window), axis=0)

    @staticmethod
    def _compute_rms(window: np.ndarray) -> np.ndarray:
        """Root Mean Square"""
        return np.sqrt(np.mean(window ** 2, axis=0))

    @staticmethod
    def _compute_var(window: np.ndarray) -> np.ndarray:
        """Variance"""
        return np.var(window, axis=0)

    @staticmethod
    def _compute_wl(window: np.ndarray) -> np.ndarray:
        """Waveform Length"""
        return np.sum(np.abs(np.diff(window, axis=0)), axis=0)

    @staticmethod
    def _compute_zc(window: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Zero Crossings"""
        centered = window - np.mean(window, axis=0, keepdims=True)
        signs = np.sign(centered)
        sign_changes = np.abs(np.diff(signs, axis=0))
        return np.sum(sign_changes > 0, axis=0).astype(np.float32)

    @staticmethod
    def _compute_ssc(window: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Slope Sign Change"""
        diff = np.diff(window, axis=0)
        signs = np.sign(diff)
        sign_changes = np.abs(np.diff(signs, axis=0))
        return np.sum(sign_changes > 0, axis=0).astype(np.float32)

    @staticmethod
    def _compute_iemg(window: np.ndarray) -> np.ndarray:
        """Integrated EMG"""
        return np.sum(np.abs(window), axis=0)

    @staticmethod
    def _compute_dasdv(window: np.ndarray) -> np.ndarray:
        """Difference Absolute Standard Deviation Value"""
        diff = np.diff(window, axis=0)
        return np.sqrt(np.mean(diff ** 2, axis=0))

    @staticmethod
    def _compute_tkeo(window: np.ndarray) -> np.ndarray:
        """Teager-Kaiser Energy Operator"""
        if len(window) < 3:
            return np.zeros(window.shape[1])
        tkeo = window[1:-1] ** 2 - window[:-2] * window[2:]
        return np.mean(np.abs(tkeo), axis=0)

    @staticmethod
    def _compute_mad(window: np.ndarray) -> np.ndarray:
        """Mean Absolute Deviation"""
        median = np.median(window, axis=0)
        return np.mean(np.abs(window - median), axis=0)

    @staticmethod
    def _compute_nle(window: np.ndarray) -> np.ndarray:
        """Normalized Logarithmic Energy"""
        energy = np.sum(window ** 2, axis=0)
        return np.log(energy / len(window) + 1e-10)


class SlidingWindowExtractor(FeatureExtractor):
    """
    スライディングウィンドウ方式の特徴量抽出器

    オーバーラップを指定可能
    """

    def __init__(
        self,
        window_size: int = 20,
        hop_size: int = 5,
        n_channels: int = 16,
        features: Optional[List[str]] = None
    ):
        super().__init__(window_size, n_channels, features)
        self.hop_size = hop_size
        self._sample_count = 0

    def update(self, sample: np.ndarray) -> Optional[np.ndarray]:
        """
        サンプルを追加し、hop_sizeごとに特徴量を返す

        Returns
        -------
        features : np.ndarray or None
            特徴量（hop_size到達時）またはNone
        """
        super().update(sample)
        self._sample_count += 1

        if self.is_ready and self._sample_count >= self.hop_size:
            self._sample_count = 0
            return self.extract()
        return None
