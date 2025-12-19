"""
Data Loader Module
==================

各種データセットの読み込みを担当
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
from dataclasses import dataclass


@dataclass
class Segment:
    """EMGセグメントデータ"""
    subject_id: int
    exercise_id: int
    movement: int
    repetition: int
    emg: np.ndarray  # (n_samples, n_channels)
    glove: np.ndarray  # (n_samples, n_joints)
    n_samples: int

    @property
    def n_channels(self) -> int:
        return self.emg.shape[1]

    @property
    def n_joints(self) -> int:
        return self.glove.shape[1]


class NinaproLoader:
    """
    Ninapro データセットローダー

    Parameters
    ----------
    filepath : str or Path
        .npzファイルのパス

    Examples
    --------
    >>> loader = NinaproLoader('ninapro_db5_segmented.npz')
    >>> print(f"セグメント数: {len(loader)}")
    >>> segment = loader[0]
    >>> print(f"EMG shape: {segment.emg.shape}")
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")

        self._load_data()

    def _load_data(self):
        """データの読み込み"""
        data = np.load(self.filepath, allow_pickle=True)
        self._raw_segments = data['segments']
        self._segments: List[Segment] = []

        for seg in self._raw_segments:
            self._segments.append(Segment(
                subject_id=int(seg['subject_id']),
                exercise_id=int(seg['exercise_id']),
                movement=int(seg['movement']),
                repetition=int(seg['repetition']),
                emg=seg['emg'].astype(np.float32),
                glove=seg['glove'].astype(np.float32),
                n_samples=int(seg['n_samples'])
            ))

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, idx: int) -> Segment:
        return self._segments[idx]

    def __iter__(self) -> Generator[Segment, None, None]:
        for seg in self._segments:
            yield seg

    def get_by_subject(self, subject_id: int) -> List[Segment]:
        """被験者IDでフィルタリング"""
        return [s for s in self._segments if s.subject_id == subject_id]

    def get_by_movement(self, movement: int) -> List[Segment]:
        """動作IDでフィルタリング"""
        return [s for s in self._segments if s.movement == movement]

    def get_by_exercise(self, exercise_id: int) -> List[Segment]:
        """課題IDでフィルタリング"""
        return [s for s in self._segments if s.exercise_id == exercise_id]

    @property
    def subjects(self) -> List[int]:
        """利用可能な被験者ID一覧"""
        return sorted(list(set(s.subject_id for s in self._segments)))

    @property
    def movements(self) -> List[int]:
        """利用可能な動作ID一覧"""
        return sorted(list(set(s.movement for s in self._segments)))

    @property
    def exercises(self) -> List[int]:
        """利用可能な課題ID一覧"""
        return sorted(list(set(s.exercise_id for s in self._segments)))

    def get_continuous_stream(self, subject_id: Optional[int] = None) -> Generator[Tuple[np.ndarray, np.ndarray, Dict], None, None]:
        """
        連続的なデータストリームを生成

        Parameters
        ----------
        subject_id : int, optional
            特定の被験者のみを対象にする場合

        Yields
        ------
        emg : np.ndarray
            EMGデータ (1, n_channels)
        glove : np.ndarray
            関節角度データ (1, n_joints)
        metadata : dict
            メタデータ
        """
        segments = self.get_by_subject(subject_id) if subject_id else self._segments

        for seg in segments:
            for i in range(seg.n_samples):
                yield (
                    seg.emg[i:i+1],
                    seg.glove[i:i+1],
                    {
                        'subject_id': seg.subject_id,
                        'movement': seg.movement,
                        'exercise_id': seg.exercise_id,
                        'sample_idx': i
                    }
                )
