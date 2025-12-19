"""
Base Device Module
==================

データソースの抽象基底クラス
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np


class DataSource(ABC):
    """
    データソースの抽象基底クラス

    すべてのデータソース（ファイル、Myo Armband等）はこのクラスを継承

    Subclasses must implement:
        - stream(): データストリームを生成
        - connect(): 接続（必要な場合）
        - disconnect(): 切断（必要な場合）
    """

    def __init__(self, n_channels: int = 16, sample_rate: float = 200.0):
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """接続状態"""
        return self._connected

    def connect(self) -> bool:
        """
        データソースに接続

        Returns
        -------
        success : bool
        """
        self._connected = True
        return True

    def disconnect(self):
        """データソースから切断"""
        self._connected = False

    @abstractmethod
    def stream(
        self,
    ) -> Generator[Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]], None, None]:
        """
        データストリームを生成

        Yields
        ------
        emg : np.ndarray
            EMGデータ (1, n_channels) または (n_channels,)
        glove : np.ndarray or None
            関節角度データ（存在する場合）
        metadata : dict
            メタデータ（タイムスタンプ、ラベル等）
        """
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False


class RealtimeDataSource(DataSource):
    """
    リアルタイムデータソースの基底クラス

    実際のハードウェアデバイス用
    """

    def __init__(self, n_channels: int = 16, sample_rate: float = 200.0):
        super().__init__(n_channels, sample_rate)
        self._device = None

    @abstractmethod
    def _init_device(self):
        """デバイス初期化"""
        pass

    @abstractmethod
    def _read_sample(self) -> np.ndarray:
        """1サンプル読み取り"""
        pass

    def connect(self) -> bool:
        try:
            self._init_device()
            self._connected = True
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def stream(
        self,
    ) -> Generator[Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]], None, None]:
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        while self._connected:
            try:
                emg = self._read_sample()
                yield emg, None, {"timestamp": None}
            except Exception as e:
                print(f"Read error: {e}")
                break
