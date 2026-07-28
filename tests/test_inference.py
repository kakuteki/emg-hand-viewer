"""推論の受け口の試験（PyTorchなしで動く範囲）"""

import numpy as np
import pytest

from emg_realtime_viz.core.inference import (
    DEFAULT_WINDOW_SIZE,
    TorchInference,
    energy_demo_model,
    wave_demo_model,
)


class FakeTensor:
    """torch.Tensorの代わり。detach/cpu/numpyだけ真似る"""

    def __init__(self, array):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class ShapePickyModel:
    """決まった形の入力しか受け付けないモデルの代役"""

    def __init__(self, accepted_ndim: int, output_dim: int = 22):
        self.accepted_ndim = accepted_ndim
        self.output_dim = output_dim
        self.seen_shapes = []

    def __call__(self, x):
        self.seen_shapes.append(tuple(x.shape))
        if len(x.shape) != self.accepted_ndim:
            raise RuntimeError(f"expected {self.accepted_ndim} dims, got {len(x.shape)}")
        return FakeTensor(np.linspace(0, 1, self.output_dim, dtype=np.float32))


def make_inference(model, **kwargs) -> TorchInference:
    """PyTorchが無い環境でも動く（入力はnumpy配列のまま渡される）"""
    return TorchInference(model, device="cpu", **kwargs)


def test_時系列モデルには窓の形で渡る():
    model = ShapePickyModel(accepted_ndim=3)
    inference = make_inference(model, smoothing=0.0)

    angles = inference(np.ones(16))

    assert inference.input_mode == "window"
    assert model.seen_shapes[0] == (1, DEFAULT_WINDOW_SIZE, 16)
    assert angles.shape == (20,)
    assert np.all((angles >= 0) & (angles <= 1))


def test_1フレーム入力のモデルにも通る():
    model = ShapePickyModel(accepted_ndim=2)
    inference = make_inference(model, smoothing=0.0)

    inference(np.ones(16))

    # 窓で失敗してからフレームに落ちる
    assert inference.input_mode == "frame"
    assert model.seen_shapes == [(1, DEFAULT_WINDOW_SIZE, 16), (1, 16)]


def test_一度決まった入力の形は覚える():
    model = ShapePickyModel(accepted_ndim=2)
    inference = make_inference(model, smoothing=0.0)

    inference(np.ones(16))
    model.seen_shapes.clear()
    inference(np.ones(16))

    assert model.seen_shapes == [(1, 16)]


class BroadcastingModel:
    """
    全結合層のように、余分な次元をそのまま通してしまうモデルの代役

    (1, 20, 16) を渡すと例外を出さずに (1, 20, 22) を返す。
    例外の有無だけで入力の形を判別すると、これを時系列モデルと
    取り違えて、最新ではなく一番古いフレームの結果を表示してしまう。
    """

    def __init__(self, output_dim: int = 22):
        self.output_dim = output_dim
        self.seen_shapes = []

    def __call__(self, x):
        self.seen_shapes.append(tuple(x.shape))
        out_shape = tuple(x.shape[:-1]) + (self.output_dim,)
        return FakeTensor(np.zeros(out_shape, dtype=np.float32))


def test_出力の形まで見て入力の形を決める():
    model = BroadcastingModel()
    inference = make_inference(model, smoothing=0.0)

    inference(np.ones(16))

    # 窓では出力が20x22個になるので退け、1フレーム入力と判定する
    assert inference.input_mode == "frame"
    assert model.seen_shapes == [(1, DEFAULT_WINDOW_SIZE, 16), (1, 16)]


def test_出力の要素数が合わなければ例外():
    model = BroadcastingModel(output_dim=7)
    inference = make_inference(model)

    with pytest.raises(RuntimeError):
        inference(np.ones(16))
    assert "要素数" in inference.last_error


def test_どの形でも通らなければ例外():
    model = ShapePickyModel(accepted_ndim=99)
    inference = make_inference(model)

    with pytest.raises(RuntimeError):
        inference(np.ones(16))
    assert inference.last_error is not None


def test_平滑化が効く():
    model = ShapePickyModel(accepted_ndim=3, output_dim=20)
    inference = make_inference(model, smoothing=0.5)

    first = inference(np.ones(16))
    second = inference(np.ones(16))

    # 出力は毎回同じなので、平滑化しても値は変わらない
    assert second == pytest.approx(first)


def test_リセットで履歴が消える():
    model = ShapePickyModel(accepted_ndim=3)
    inference = make_inference(model)

    inference(np.ones(16))
    inference.reset()

    assert len(inference._buffer) == 0


@pytest.mark.parametrize("factory", [energy_demo_model, wave_demo_model])
def test_デモ推論は20次元の0から1を返す(factory):
    model = factory()
    for _ in range(5):
        angles = model(np.random.randn(16) * 50)
        assert angles.shape == (20,)
        assert np.all((angles >= 0) & (angles <= 1))


def test_エネルギーデモは強い入力ほど曲がる():
    weak = energy_demo_model(smoothing=0.0)(np.zeros(16))
    strong = energy_demo_model(smoothing=0.0)(np.full(16, 100.0))
    assert np.all(strong >= weak)
    assert np.any(strong > weak)
