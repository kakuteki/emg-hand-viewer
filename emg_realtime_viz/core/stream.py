"""
Data Stream Module
==================

リアルタイムデータストリーミング処理
"""

import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

import numpy as np


class StreamState(Enum):
    """ストリームの状態"""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class StreamData:
    """ストリームデータパケット"""

    timestamp: float
    emg: np.ndarray
    glove: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class DataStream:
    """
    リアルタイムデータストリーム管理

    データソースからのデータを非同期で処理し、
    コールバックまたはキューを通じて提供

    Parameters
    ----------
    source : DataSource
        データソース
    buffer_size : int
        バッファサイズ
    playback_speed : float
        再生速度（1.0 = リアルタイム）

    Examples
    --------
    >>> stream = DataStream(source, buffer_size=100)
    >>> stream.on_data = lambda data: print(data.emg.shape)
    >>> stream.start()
    >>> time.sleep(10)
    >>> stream.stop()
    """

    def __init__(
        self,
        source,  # DataSource
        buffer_size: int = 1000,
        playback_speed: float = 1.0,
        sample_rate: float = 200.0,
    ):
        self.source = source
        self.buffer_size = buffer_size
        self.playback_speed = playback_speed
        self.sample_rate = sample_rate

        self._state = StreamState.STOPPED
        self._thread: Optional[threading.Thread] = None
        self._data_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        # コールバック
        self.on_data: Optional[Callable[[StreamData], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_state_change: Optional[Callable[[StreamState], None]] = None

        # 統計
        self._sample_count = 0
        self._start_time: Optional[float] = None

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def elapsed_time(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def start(self):
        """ストリーミング開始"""
        if self._state == StreamState.RUNNING:
            return

        self._stop_event.clear()
        self._pause_event.set()  # not paused
        self._sample_count = 0
        self._start_time = time.time()

        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

        self._set_state(StreamState.RUNNING)

    def stop(self):
        """ストリーミング停止"""
        self._stop_event.set()
        self._pause_event.set()  # unpause to allow thread to exit

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._set_state(StreamState.STOPPED)

    def pause(self):
        """一時停止"""
        if self._state == StreamState.RUNNING:
            self._pause_event.clear()
            self._set_state(StreamState.PAUSED)

    def resume(self):
        """再開"""
        if self._state == StreamState.PAUSED:
            self._pause_event.set()
            self._set_state(StreamState.RUNNING)

    def get_data(self, timeout: float = 0.1) -> Optional[StreamData]:
        """
        キューからデータを取得

        Parameters
        ----------
        timeout : float
            タイムアウト（秒）

        Returns
        -------
        data : StreamData or None
        """
        try:
            return self._data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _set_state(self, state: StreamState):
        self._state = state
        if self.on_state_change:
            self.on_state_change(state)

    def _stream_loop(self):
        """ストリーミングループ（別スレッド）"""
        interval = 1.0 / (self.sample_rate * self.playback_speed)

        try:
            for emg, glove, metadata in self.source.stream():
                # 停止チェック
                if self._stop_event.is_set():
                    break

                # 一時停止チェック
                self._pause_event.wait()

                # データパケット作成
                data = StreamData(timestamp=time.time(), emg=emg, glove=glove, metadata=metadata)

                # コールバック呼び出し
                if self.on_data:
                    self.on_data(data)

                # キューに追加（満杯なら古いものを破棄）
                try:
                    self._data_queue.put_nowait(data)
                except queue.Full:
                    try:
                        self._data_queue.get_nowait()
                        self._data_queue.put_nowait(data)
                    except queue.Empty:
                        pass

                self._sample_count += 1

                # レート制御
                time.sleep(interval)

        except Exception as e:
            if self.on_error:
                self.on_error(e)
            else:
                raise

        self._set_state(StreamState.STOPPED)


class FeatureStream(DataStream):
    """
    特徴量付きデータストリーム

    EMGデータに加えて特徴量も計算して提供
    """

    def __init__(self, source, feature_extractor, **kwargs):
        super().__init__(source, **kwargs)
        self.feature_extractor = feature_extractor

    def _stream_loop(self):
        """特徴量計算付きストリーミングループ"""
        interval = 1.0 / (self.sample_rate * self.playback_speed)

        try:
            for emg, glove, metadata in self.source.stream():
                if self._stop_event.is_set():
                    break

                self._pause_event.wait()

                # 特徴量抽出
                self.feature_extractor.update(emg.flatten())
                features = None
                if self.feature_extractor.is_ready:
                    features = self.feature_extractor.extract()

                data = StreamData(
                    timestamp=time.time(),
                    emg=emg,
                    glove=glove,
                    features=features,
                    metadata=metadata,
                )

                if self.on_data:
                    self.on_data(data)

                try:
                    self._data_queue.put_nowait(data)
                except queue.Full:
                    try:
                        self._data_queue.get_nowait()
                        self._data_queue.put_nowait(data)
                    except queue.Empty:
                        pass

                self._sample_count += 1
                time.sleep(interval)

        except Exception as e:
            if self.on_error:
                self.on_error(e)
            else:
                raise

        self._set_state(StreamState.STOPPED)
