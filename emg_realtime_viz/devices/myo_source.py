"""
Myo Armband Data Source
=======================

Myo Armbandからのリアルタイムデータ取得

対応ライブラリ:
    - pyomyo: pip install pyomyo
    - myo-python: pip install myo-python

Notes
-----
Myo Connectが起動している必要があります。
Windows: Myo Connect をインストールして起動
"""

import numpy as np
from typing import Generator, Tuple, Dict, Any, Optional, Callable
from collections import deque
import time
import threading
from queue import Queue, Empty

from .base import RealtimeDataSource

# Myo ライブラリのインポート試行
MYO_BACKEND = None

try:
    from pyomyo import Myo, emg_mode
    MYO_BACKEND = 'pyomyo'
except ImportError:
    pass

if MYO_BACKEND is None:
    try:
        import myo
        MYO_BACKEND = 'myo-python'
    except ImportError:
        pass


class MyoDataSource(RealtimeDataSource):
    """
    Myo Armbandデータソース

    Parameters
    ----------
    n_channels : int
        EMGチャンネル数 (Myoは8チャンネル)
    sample_rate : float
        サンプリングレート (Myoは約200Hz)
    buffer_size : int
        内部バッファサイズ
    emg_mode : str
        EMGモード ('filtered' or 'raw')

    Examples
    --------
    >>> source = MyoDataSource()
    >>> with source:
    ...     for emg, _, meta in source.stream():
    ...         print(emg.shape)  # (1, 8)

    Notes
    -----
    使用前に以下が必要:
    1. Myo Connect のインストールと起動
    2. pyomyo または myo-python のインストール
       pip install pyomyo
    """

    def __init__(
        self,
        n_channels: int = 8,
        sample_rate: float = 200.0,
        buffer_size: int = 1000,
        emg_mode: str = 'filtered'
    ):
        super().__init__(n_channels, sample_rate)
        self.buffer_size = buffer_size
        self._emg_mode = emg_mode

        # データキュー
        self._data_queue: Queue = Queue(maxsize=buffer_size)
        self._imu_data: Dict[str, np.ndarray] = {}

        # Myoオブジェクト
        self._myo = None
        self._listener = None
        self._hub = None

        # スレッド
        self._read_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 統計
        self._sample_count = 0
        self._start_time = None

    def _init_device(self):
        """デバイス初期化"""
        if MYO_BACKEND is None:
            raise ImportError(
                "Myoライブラリが見つかりません。以下のいずれかをインストールしてください:\n"
                "  pip install pyomyo\n"
                "  pip install myo-python"
            )

        if MYO_BACKEND == 'pyomyo':
            self._init_pyomyo()
        elif MYO_BACKEND == 'myo-python':
            self._init_myo_python()

    def _init_pyomyo(self):
        """pyomyoでの初期化"""
        from pyomyo import Myo, emg_mode

        print("Myo Armbandに接続中 (pyomyo)...")

        # EMGモード設定
        if self._emg_mode == 'raw':
            mode = emg_mode.RAW
        else:
            mode = emg_mode.FILTERED

        self._myo = Myo(mode=mode)
        self._myo.connect()

        # コールバック設定
        self._myo.add_emg_handler(self._on_emg_pyomyo)
        self._myo.add_imu_handler(self._on_imu_pyomyo)

        print(f"Myo接続完了 (Backend: pyomyo, Mode: {self._emg_mode})")

    def _init_myo_python(self):
        """myo-pythonでの初期化"""
        import myo

        print("Myo Armbandに接続中 (myo-python)...")

        myo.init()
        self._hub = myo.Hub()

        # リスナー作成
        self._listener = _MyoPythonListener(self._data_queue, self._imu_data)

        print("Myo接続完了 (Backend: myo-python)")

    def _on_emg_pyomyo(self, emg, moving):
        """pyomyo EMGコールバック"""
        try:
            emg_array = np.array(emg, dtype=np.float32).reshape(1, -1)
            timestamp = time.time()

            if not self._data_queue.full():
                self._data_queue.put_nowait((emg_array, timestamp))
                self._sample_count += 1
        except Exception as e:
            print(f"EMG callback error: {e}")

    def _on_imu_pyomyo(self, quat, acc, gyro):
        """pyomyo IMUコールバック"""
        self._imu_data['quaternion'] = np.array(quat)
        self._imu_data['acceleration'] = np.array(acc)
        self._imu_data['gyroscope'] = np.array(gyro)

    def _read_sample(self) -> np.ndarray:
        """1サンプル読み取り"""
        # pyomyoの場合はrun()を呼ぶ必要がある
        if MYO_BACKEND == 'pyomyo' and self._myo:
            self._myo.run()

        try:
            emg, timestamp = self._data_queue.get(timeout=0.1)
            return emg
        except Empty:
            # データがない場合はゼロを返す
            return np.zeros((1, self.n_channels), dtype=np.float32)

    def connect(self) -> bool:
        """接続"""
        try:
            self._init_device()
            self._connected = True
            self._stop_event.clear()
            self._start_time = time.time()
            self._sample_count = 0

            # myo-pythonの場合はバックグラウンドスレッド開始
            if MYO_BACKEND == 'myo-python' and self._hub:
                self._read_thread = threading.Thread(target=self._myo_python_loop)
                self._read_thread.daemon = True
                self._read_thread.start()

            return True
        except Exception as e:
            print(f"Myo接続エラー: {e}")
            return False

    def _myo_python_loop(self):
        """myo-python用のバックグラウンドループ"""
        import myo
        while not self._stop_event.is_set():
            self._hub.run(self._listener.on_event, 500)

    def disconnect(self):
        """切断"""
        self._stop_event.set()
        self._connected = False

        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)

        if MYO_BACKEND == 'pyomyo' and self._myo:
            try:
                self._myo.disconnect()
            except:
                pass
            self._myo = None

        if MYO_BACKEND == 'myo-python' and self._hub:
            try:
                self._hub.shutdown()
            except:
                pass
            self._hub = None

        print("Myo切断完了")

    def stream(self) -> Generator[Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]], None, None]:
        """データストリーム生成"""
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        while self._connected:
            try:
                # pyomyoの場合は明示的にrun()を呼ぶ
                if MYO_BACKEND == 'pyomyo' and self._myo:
                    self._myo.run()

                # キューからデータ取得
                try:
                    emg, timestamp = self._data_queue.get(timeout=0.05)
                except Empty:
                    continue

                # メタデータ
                elapsed = time.time() - self._start_time if self._start_time else 0
                metadata = {
                    'timestamp': timestamp,
                    'elapsed': elapsed,
                    'sample_count': self._sample_count,
                    'backend': MYO_BACKEND,
                    'imu': self._imu_data.copy() if self._imu_data else None
                }

                yield emg, None, metadata

            except Exception as e:
                print(f"Stream error: {e}")
                break

    def vibrate(self, duration: str = 'short'):
        """
        Myoを振動させる

        Parameters
        ----------
        duration : str
            'short', 'medium', 'long'
        """
        if MYO_BACKEND == 'pyomyo' and self._myo:
            self._myo.vibrate(duration)

    @property
    def actual_sample_rate(self) -> float:
        """実際のサンプリングレート"""
        if self._start_time and self._sample_count > 0:
            elapsed = time.time() - self._start_time
            return self._sample_count / elapsed if elapsed > 0 else 0
        return 0

    @property
    def imu_data(self) -> Dict[str, np.ndarray]:
        """最新のIMUデータ"""
        return self._imu_data.copy()


class _MyoPythonListener:
    """myo-python用のイベントリスナー"""

    def __init__(self, data_queue: Queue, imu_data: Dict):
        self._queue = data_queue
        self._imu_data = imu_data
        self._sample_count = 0

    def on_event(self, event):
        """イベントハンドラ"""
        import myo

        if event.type == myo.EventType.emg:
            emg = np.array(event.emg, dtype=np.float32).reshape(1, -1)
            timestamp = time.time()

            if not self._queue.full():
                try:
                    self._queue.put_nowait((emg, timestamp))
                    self._sample_count += 1
                except:
                    pass

        elif event.type == myo.EventType.orientation:
            self._imu_data['quaternion'] = np.array([
                event.orientation.x,
                event.orientation.y,
                event.orientation.z,
                event.orientation.w
            ])
            self._imu_data['acceleration'] = np.array([
                event.acceleration.x,
                event.acceleration.y,
                event.acceleration.z
            ])
            self._imu_data['gyroscope'] = np.array([
                event.gyroscope.x,
                event.gyroscope.y,
                event.gyroscope.z
            ])


class SimulatedMyoSource(RealtimeDataSource):
    """
    シミュレーションMyoデータソース

    実際のMyoがない場合のテスト用

    Parameters
    ----------
    n_channels : int
        チャンネル数
    sample_rate : float
        サンプリングレート
    noise_level : float
        ノイズレベル
    """

    def __init__(
        self,
        n_channels: int = 8,
        sample_rate: float = 200.0,
        noise_level: float = 0.1
    ):
        super().__init__(n_channels, sample_rate)
        self.noise_level = noise_level
        self._sample_idx = 0
        self._start_time = None

    def _init_device(self):
        """デバイス初期化（シミュレーション）"""
        print("シミュレーションMyoを初期化中...")
        self._sample_idx = 0
        self._start_time = time.time()
        print("シミュレーションMyo準備完了")

    def _read_sample(self) -> np.ndarray:
        """シミュレーションデータ生成"""
        t = self._sample_idx / self.sample_rate

        # 複数の周波数成分を持つ疑似EMG信号
        emg = np.zeros(self.n_channels)
        for ch in range(self.n_channels):
            # 各チャンネルで異なる周波数
            freq1 = 10 + ch * 5
            freq2 = 50 + ch * 10

            signal = (
                0.3 * np.sin(2 * np.pi * freq1 * t + ch) +
                0.2 * np.sin(2 * np.pi * freq2 * t + ch * 0.5) +
                self.noise_level * np.random.randn()
            )

            # 間欠的な筋活動をシミュレート
            if (self._sample_idx // 200) % 3 == ch % 3:
                signal *= 3

            emg[ch] = signal

        self._sample_idx += 1

        # サンプリングレートに合わせて待機
        expected_time = self._sample_idx / self.sample_rate
        elapsed = time.time() - self._start_time
        if elapsed < expected_time:
            time.sleep(expected_time - elapsed)

        return emg.reshape(1, -1).astype(np.float32)

    def stream(self) -> Generator[Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]], None, None]:
        """ストリーム生成"""
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        while self._connected:
            emg = self._read_sample()

            metadata = {
                'timestamp': time.time(),
                'sample_idx': self._sample_idx,
                'simulated': True
            }

            yield emg, None, metadata


def get_myo_source(simulated: bool = False, **kwargs):
    """
    Myoデータソースを取得

    Parameters
    ----------
    simulated : bool
        シミュレーションモードを使用するか
    **kwargs
        データソースに渡すパラメータ

    Returns
    -------
    DataSource
        MyoDataSource or SimulatedMyoSource
    """
    if simulated or MYO_BACKEND is None:
        if MYO_BACKEND is None:
            print("Myoライブラリが見つかりません。シミュレーションモードを使用します。")
        return SimulatedMyoSource(**kwargs)
    else:
        return MyoDataSource(**kwargs)
