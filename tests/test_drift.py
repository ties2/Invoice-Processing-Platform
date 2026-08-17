from src.monitoring.drift_detector import detect_drift, population_stability_index


def test_no_drift_for_similar_distributions():
    base = [0.9, 0.92, 0.95, 0.88, 0.91] * 20
    curr = [0.9, 0.93, 0.94, 0.89, 0.92] * 20
    r = detect_drift(base, curr, {"A", "B"}, ["A", "B", "A"])
    assert not r.drift_detected


def test_confidence_drift_flags():
    base = [0.95] * 100
    curr = [0.55] * 100
    r = detect_drift(base, curr, {"A"}, ["A"])
    assert r.drift_detected
    assert r.confidence_psi > 0.25


def test_new_supplier_surge_flags():
    base = [0.9] * 50
    curr = [0.9] * 50
    r = detect_drift(base, curr, {"A"}, ["A", "New1", "New2", "New3"])
    assert r.drift_detected
    assert r.new_supplier_rate > 0.30


def test_psi_zero_for_identical():
    d = [0.1, 0.3, 0.5, 0.7, 0.9] * 10
    assert population_stability_index(d, d) < 0.01
