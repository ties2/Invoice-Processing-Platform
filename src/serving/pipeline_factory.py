"""Pipeline factory (IPP-009).

The one place that decides how the extraction pipeline is wired: which extractor
is the champion, whether a challenger runs in shadow/canary, and the known-
supplier registry. Swapping in a new model or turning on a shadow experiment is a
change here only — the API and service never move.
"""

from __future__ import annotations

from functools import lru_cache

from src.pipeline.experiment import ExperimentMode, ExtractionExperiment
from src.pipeline.extractor import RuleBasedExtractor
from src.pipeline.pipeline import ExtractionPipeline

# In production this would be loaded from a supplier master table; seeded here.
KNOWN_SUPPLIERS = {"KPN B.V.", "ACME Trading B.V."}


@lru_cache(maxsize=1)
def get_pipeline() -> ExtractionPipeline:
    champion = RuleBasedExtractor(known_suppliers=KNOWN_SUPPLIERS)
    # Default: champion only. To shadow-test a challenger, build it here and set
    # mode=ExperimentMode.SHADOW (or CANARY with canary_pct=...).
    experiment = ExtractionExperiment(champion=champion, mode=ExperimentMode.CHAMPION_ONLY)
    return ExtractionPipeline(experiment=experiment)
