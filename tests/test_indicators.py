import numpy as np
from gpu_rl_trading.env.indicators import build_feature_matrix


def test_build_feature_matrix_shape_and_finite():
    T = 500
    rng = np.random.RandomState(0)
    ohlcv = (rng.rand(T, 5).astype(np.float32) + 1.0)
    mat = build_feature_matrix(ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4])
    assert mat.shape[0] == T
    assert mat.shape[1] == 27
    assert np.isfinite(mat).all()
