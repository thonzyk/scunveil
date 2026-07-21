import numpy as np
from scipy.sparse import csr_matrix

from scunveil._data_operations import logits_to_CPM, simple_scipy_norm_x


def test_log10_cpm_is_stable_for_extreme_logits():
    result = logits_to_CPM(np.array([[0.0, -1000.0]], dtype=np.float32))

    assert np.isfinite(result).all()
    assert result[0, 0] == 6.0
    assert result[0, 1] < -400.0


def test_sparse_log1p_does_not_overflow_at_large_counts():
    matrix = csr_matrix(np.array([[0.0, 1_000_000.0]], dtype=np.float64))
    result = simple_scipy_norm_x(matrix).numpy()

    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[0, 1], np.log1p(1_000_000), rtol=1e-6)
