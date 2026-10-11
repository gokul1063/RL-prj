import numpy as np

from cluster_rl.sim.failure_model import FailureModel


def test_fit_and_predict_range(instances):
    fm = FailureModel(beta=0.5).fit(instances)
    assert fm.clf is not None
    assert 0.0 < fm.prior_ < 1.0
    p = fm.base_probability(0.05, 0.05, 5, 1)
    assert 0.0 <= p <= 1.0


def test_high_priority_less_risky(instances):
    fm = FailureModel().fit(instances)
    p_free = fm.base_probability(0.05, 0.05, 0, 1)
    p_prod = fm.base_probability(0.05, 0.05, 10, 1)
    assert p_free >= p_prod


def test_save_load_roundtrip(instances, tmp_path):
    fm = FailureModel().fit(instances)
    path = tmp_path / "fm.pkl"
    fm.save(path)
    loaded = FailureModel.load(path)
    assert np.isclose(
        loaded.base_probability(0.03, 0.04, 9, 2),
        fm.base_probability(0.03, 0.04, 9, 2),
    )
