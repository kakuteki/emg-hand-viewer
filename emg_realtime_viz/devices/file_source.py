"""
File-based Data Sources
=======================

ファイルからのデータ再生
"""

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from ..core.data_loader import NinaproLoader, Segment
from .base import DataSource


class NinaproDataSource(DataSource):
    """
    Ninapro データセットからのデータソース

    Parameters
    ----------
    filepath : str
        .npzファイルのパス
    subject_ids : list of int, optional
        使用する被験者ID
    movements : list of int, optional
        使用する動作ID
    loop : bool
        データ終了時にループするか

    Examples
    --------
    >>> source = NinaproDataSource('ninapro_db5_segmented.npz')
    >>> for emg, glove, meta in source.stream():
    ...     print(emg.shape)
    """

    def __init__(
        self,
        filepath: str,
        subject_ids: Optional[List[int]] = None,
        movements: Optional[List[int]] = None,
        loop: bool = True,
        n_channels: int = 16,
        sample_rate: float = 200.0,
    ):
        super().__init__(n_channels, sample_rate)
        self.filepath = Path(filepath)
        self.subject_ids = subject_ids
        self.movements = movements
        self.loop = loop

        self._loader: Optional[NinaproLoader] = None
        self._segments: List[Segment] = []
        self._current_segment_idx = 0
        self._current_sample_idx = 0

    def connect(self) -> bool:
        try:
            self._loader = NinaproLoader(self.filepath)

            # フィルタリング
            self._segments = list(self._loader)

            if self.subject_ids:
                self._segments = [s for s in self._segments if s.subject_id in self.subject_ids]

            if self.movements:
                self._segments = [s for s in self._segments if s.movement in self.movements]

            if not self._segments:
                raise ValueError("No segments match the filter criteria")

            self._current_segment_idx = 0
            self._current_sample_idx = 0
            self._connected = True
            return True

        except Exception as e:
            print(f"Failed to load data: {e}")
            return False

    def disconnect(self):
        self._loader = None
        self._segments = []
        self._connected = False

    def stream(
        self,
    ) -> Generator[Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]], None, None]:
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        # 位置を進めてからyieldする。yieldの後で進めると、
        # 中断中にreset()されても次の再開時に上書きしてしまい、
        # 画面のResetボタンを押しても先頭に戻らない。
        skipped = 0

        while True:
            if self._current_segment_idx >= len(self._segments):
                if not self.loop:
                    break
                self._current_segment_idx = 0
                self._current_sample_idx = 0

            segment = self._segments[self._current_segment_idx]

            if self._current_sample_idx >= segment.n_samples:
                self._current_sample_idx = 0
                self._current_segment_idx += 1

                # 中身が空のセグメントばかりだと、ループ指定のときに
                # 1件も返さないまま回り続けてしまうので打ち切る
                skipped += 1
                if skipped > len(self._segments):
                    return
                continue

            sample_idx = self._current_sample_idx
            emg = segment.emg[sample_idx : sample_idx + 1]
            glove = segment.glove[sample_idx : sample_idx + 1]

            metadata = {
                "subject_id": segment.subject_id,
                "movement": segment.movement,
                "exercise_id": segment.exercise_id,
                "repetition": segment.repetition,
                "segment_idx": self._current_segment_idx,
                "sample_idx": sample_idx,
                "total_samples": segment.n_samples,
            }

            self._current_sample_idx = sample_idx + 1
            skipped = 0

            yield emg, glove, metadata

    def reset(self):
        """ストリームを先頭にリセット"""
        self._current_segment_idx = 0
        self._current_sample_idx = 0

    @property
    def total_segments(self) -> int:
        return len(self._segments)

    @property
    def current_progress(self) -> Tuple[int, int, int, int]:
        """現在の進捗 (segment_idx, n_segments, sample_idx, n_samples)"""
        if not self._segments:
            return (0, 0, 0, 0)

        # 最後まで再生し終えた直後はセグメント番号が末尾を越えているので丸める
        seg_idx = min(self._current_segment_idx, len(self._segments) - 1)
        seg = self._segments[seg_idx]
        return (
            seg_idx,
            len(self._segments),
            self._current_sample_idx,
            seg.n_samples,
        )


class FilePlaybackSource(DataSource):
    """
    汎用ファイル再生データソース

    numpy配列ファイルからの再生

    Parameters
    ----------
    emg_data : np.ndarray
        EMGデータ (n_samples, n_channels)
    glove_data : np.ndarray, optional
        関節角度データ (n_samples, n_joints)
    loop : bool
        ループ再生
    """

    def __init__(
        self,
        emg_data: np.ndarray,
        glove_data: Optional[np.ndarray] = None,
        loop: bool = True,
        sample_rate: float = 200.0,
    ):
        n_channels = emg_data.shape[1] if emg_data.ndim > 1 else 1
        super().__init__(n_channels, sample_rate)

        self.emg_data = emg_data
        self.glove_data = glove_data
        self.loop = loop
        self._current_idx = 0

    def connect(self) -> bool:
        self._current_idx = 0
        self._connected = True
        return True

    def stream(
        self,
    ) -> Generator[Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]], None, None]:
        n_samples = len(self.emg_data)
        if n_samples == 0:
            return

        # NinaproDataSourceと同じ理由で、進めてからyieldする
        while True:
            if self._current_idx >= n_samples:
                if not self.loop:
                    break
                self._current_idx = 0

            idx = self._current_idx
            emg = self.emg_data[idx : idx + 1]
            glove = None
            if self.glove_data is not None:
                glove = self.glove_data[idx : idx + 1]

            metadata = {
                "sample_idx": idx,
                "total_samples": n_samples,
                "progress": idx / n_samples,
            }

            self._current_idx = idx + 1

            yield emg, glove, metadata

    def reset(self):
        self._current_idx = 0

    @classmethod
    def from_file(cls, filepath: str, **kwargs) -> "FilePlaybackSource":
        """ファイルから作成"""
        data = np.load(filepath)
        if isinstance(data, np.ndarray):
            return cls(emg_data=data, **kwargs)
        else:
            # .npzファイル
            emg = data.get("emg", data.get("data", None))
            glove = data.get("glove", None)
            return cls(emg_data=emg, glove_data=glove, **kwargs)
