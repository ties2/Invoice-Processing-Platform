from src.pipeline.ocr import SimulatedOCR
from src.pipeline.extractor import RuleBasedExtractor, LegacyExtractor
from src.pipeline.experiment import ExtractionExperiment, ExperimentMode


def _ocr(clean_invoice_bytes):
    return SimulatedOCR(noise=0.0).recognize(clean_invoice_bytes)


def test_shadow_serves_champion_but_surfaces_disagreement(clean_invoice_bytes):
    exp = ExtractionExperiment(LegacyExtractor(), RuleBasedExtractor(), mode=ExperimentMode.SHADOW)
    served, report = exp.run(_ocr(clean_invoice_bytes), routing_key="k")
    assert report.served_by == "champion"          # shadow never serves challenger
    assert report.is_corner_case                    # legacy vs modern disagree
    assert any(d.field_name == "total" for d in report.disagreements)


def test_champion_only_has_no_challenger(clean_invoice_bytes):
    exp = ExtractionExperiment(RuleBasedExtractor(), mode=ExperimentMode.CHAMPION_ONLY)
    _, report = exp.run(_ocr(clean_invoice_bytes))
    assert report.challenger_version is None


def test_canary_is_deterministic_and_roughly_proportional():
    exp = ExtractionExperiment(LegacyExtractor(), RuleBasedExtractor(),
                               mode=ExperimentMode.CANARY, canary_pct=30)
    # deterministic: same key -> same arm
    assert exp._canary_selects_challenger("supplier-x") == exp._canary_selects_challenger("supplier-x")
    # proportional over many keys
    share = sum(exp._canary_selects_challenger(f"k{i}") for i in range(3000)) / 3000
    assert 0.25 < share < 0.35


def test_shadow_requires_challenger():
    import pytest
    with pytest.raises(ValueError):
        ExtractionExperiment(RuleBasedExtractor(), mode=ExperimentMode.SHADOW)
