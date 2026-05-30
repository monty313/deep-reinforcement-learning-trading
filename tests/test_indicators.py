"""Tests for gpu_rl_trading.env.indicators."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
from gpu_rl_trading.env.indicators import (
    sma, ema, atr, rsi, cci, bb_bands, build_feature_matrix,
    _rolling_mean, _rolling_std,
)


class TestRollingMean:
    def test_simple_mean(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = _rolling_mean(x, 2)
        assert np.isnan(result[0])
        assert result[1] == 1.5
        assert result[2] == 2.5


class TestRollingStd:
    def test_simple_std(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = _rolling_std(x, 2)
        assert np.isnan(result[0])
        # std([1, 2]) = 0.5
        assert np.isclose(result[1], 0.5, atol=1e-5)


class TestSMA:
    def test_sma_length(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = sma(x, 2)
        assert len(result) == len(x)
        assert np.isnan(result[0])
        assert result[1] == 1.5


class TestEMA:
    def test_ema_length(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = ema(x, 2)
        assert len(result) == len(x)
        assert not np.isnan(result[1])


class TestATR:
    def test_atr_length(self):
        h = np.array([1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float32)
        l = np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float32)
        c = np.array([1.05, 1.15, 1.25, 1.35, 1.45], dtype=np.float32)
        result = atr(h, l, c, 2)
        assert len(result) == len(h)
        assert not np.isnan(result[-1])


class TestRSI:
    def test_rsi_range(self):
        c = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.1], dtype=np.float32)
        result = rsi(c, 2)
        assert len(result) == len(c)
        assert np.all((result >= 0) | np.isnan(result))
        assert np.all((result <= 100) | np.isnan(result))


class TestCCI:
    def test_cci_length(self):
        h = np.array([1.1, 1.2, 1.3, 1.4, 1.5] * 5, dtype=np.float32)
        l = np.array([1.0, 1.1, 1.2, 1.3, 1.4] * 5, dtype=np.float32)
        c = np.array([1.05, 1.15, 1.25, 1.35, 1.45] * 5, dtype=np.float32)
        result = cci(h, l, c, 3)
        assert len(result) == len(h)


class TestBBands:
    def test_bb_bands_shape(self):
        c = np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float32)
        upper, mid, lower = bb_bands(c, 2)
        assert len(upper) == len(c)
        assert len(mid) == len(c)
        assert len(lower) == len(c)
        # upper should be >= mid >= lower (ignoring NaNs)
        mask = ~np.isnan(upper) & ~np.isnan(mid) & ~np.isnan(lower)
        assert np.all(upper[mask] >= mid[mask])
        assert np.all(mid[mask] >= lower[mask])


class TestBuildFeatureMatrix:
    def test_output_shape(self):
        T = 500
        o = np.random.rand(T).astype(np.float32) + 1.0
        h = o + np.abs(np.random.rand(T).astype(np.float32)) * 0.01
        l = o - np.abs(np.random.rand(T).astype(np.float32)) * 0.01
        c = (h + l) / 2 + np.random.rand(T).astype(np.float32) * 0.005
        v = np.random.rand(T).astype(np.float32) * 1000
        
        features = build_feature_matrix(o, h, l, c, v)
        assert features.shape == (T, 27), f"Expected (500, 27), got {features.shape}"
        assert features.dtype == np.float32
        
    def test_no_inf_nan_in_output(self):
        T = 500
        o = np.random.rand(T).astype(np.float32) + 1.0
        h = o + np.abs(np.random.rand(T).astype(np.float32)) * 0.01
        l = o - np.abs(np.random.rand(T).astype(np.float32)) * 0.01
        c = (h + l) / 2
        v = np.random.rand(T).astype(np.float32) * 1000
        
        features = build_feature_matrix(o, h, l, c, v)
        # build_feature_matrix converts NaN to 0, so no NaNs/Infs should remain
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
