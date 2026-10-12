"""Shadow-mode A/B: rule-based champion vs LLM challenger on real receipts.

Runs the production `ExtractionExperiment` in SHADOW mode over real SROIE
invoices: the rule-based extractor serves (as in prod), the LLM extractor runs
in the shadow on the same pages, and we score BOTH against ground truth to see
whether the challenger is worth promoting. Disagreements are surfaced as
candidate corner cases — exactly the IPP-004 feedback loop.

    export IPP_DATASET_ROOT=/path/to/SROIE/data

    # with your own API key -> live LLM calls:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/run_experiment.py --n 15

    # no key / reproducible benchmark -> replay the recorded cache:
    python scripts/run_experiment.py --n 15 --cache scripts/llm_shadow_cache.json

The two paths run the *identical* pipeline code; only the LLM client differs
(real `anthropic.Anthropic()` vs `RecordedLLMClient`).
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


import argparse

from src.datasets.sroie import SroieDataset, score_sample
from src.pipeline.ocr_tesseract import TesseractOCR
from src.pipeline.extractor import RuleBasedExtractor
from src.pipeline.extractor_llm import LLMExtractor, RecordedLLMClient
from src.pipeline.experiment import ExtractionExperiment, ExperimentMode


def _acc_table(hits, totals, label):
    print(f"\n{label}")
    for f in sorted(totals):
        n, t = hits.get(f, 0), totals[f]
        bar = "#" * round(12 * n / t) if t else ""
        print(f"  {f:16s} {n:2d}/{t:<2d} {n/t:4.0%}  {bar}")
    micro_h, micro_t = sum(hits.values()), sum(totals.values())
    print(f"  {'OVERALL':16s} {micro_h:2d}/{micro_t:<2d} {micro_h/micro_t:4.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--cache", default=None, help="recorded LLM responses (offline mode)")
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--plot", default=None, help="save a champion-vs-challenger bar chart here")
    args = ap.parse_args()

    ds = SroieDataset(args.root)
    ocr_engine = TesseractOCR(lang=args.lang)

    champion = RuleBasedExtractor()
    if args.cache:
        challenger = LLMExtractor(client=RecordedLLMClient.from_file(args.cache))
    else:
        challenger = LLMExtractor()  # real anthropic.Anthropic() from env key

    exp = ExtractionExperiment(
        champion=champion, challenger=challenger, mode=ExperimentMode.SHADOW,
    )

    champ_hits, chall_hits, totals = {}, {}, {}
    agreements, corner_cases, n = 0.0, 0, 0

    for sid in ds.ids()[: args.n]:
        sample = ds.load(sid)
        ocr = ocr_engine.recognize(sample.image_bytes)
        served, report = exp.run(ocr, routing_key=sid)   # SHADOW: served == champion

        champ_preds = served
        chall_preds = challenger.extract(ocr)            # the shadow arm's output
        n += 1
        agreements += report.agreement_rate
        corner_cases += int(report.is_corner_case)

        for s in score_sample(champ_preds, sample.ground_truth):
            totals[s.field_name] = totals.get(s.field_name, 0) + 1
            champ_hits[s.field_name] = champ_hits.get(s.field_name, 0) + int(s.correct)
        for s in score_sample(chall_preds, sample.ground_truth):
            chall_hits[s.field_name] = chall_hits.get(s.field_name, 0) + int(s.correct)

    print(f"Shadow experiment over {n} real receipts "
          f"(champion={champion.version}, challenger={challenger.version})")
    print(f"mean field-agreement champion/challenger: {agreements/n:.0%}")
    print(f"corner cases auto-surfaced for labelling : {corner_cases}/{n}")
    _acc_table(champ_hits, totals, "CHAMPION  (rule-based, serving)")
    _acc_table(chall_hits, totals, "CHALLENGER(LLM, shadow)")

    ch, cl, tot = sum(champ_hits.values()), sum(chall_hits.values()), sum(totals.values())
    delta = (cl - ch) / tot
    verdict = "PROMOTE challenger" if delta > 0.05 else "keep champion"
    print(f"\nVerdict: challenger {cl/tot:.0%} vs champion {ch/tot:.0%} "
          f"({delta:+.0%} pts) -> {verdict}")

    if args.plot:
        _plot(champ_hits, chall_hits, totals, n,
              champion.version, challenger.version, args.plot)
        print(f"chart saved -> {args.plot}")


def _plot(champ_hits, chall_hits, totals, n, champ_v, chall_v, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = sorted(totals) + ["overall"]
    def acc(h):
        vals = [h.get(f, 0) / totals[f] for f in sorted(totals)]
        return vals + [sum(h.values()) / sum(totals.values())]
    champ, chall = acc(champ_hits), acc(chall_hits)

    x = range(len(fields)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    b1 = ax.bar([i - w/2 for i in x], champ, w, label=f"Champion · rule-based ({champ_v})", color="#94a3b8")
    b2 = ax.bar([i + w/2 for i in x], chall, w, label=f"Challenger · LLM ({chall_v})", color="#2563eb")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{b.get_height():.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Extraction accuracy vs ground truth")
    ax.set_xticks(list(x)); ax.set_xticklabels([f.replace("_", " ") for f in fields])
    ax.set_title(f"Shadow mode: LLM challenger vs rule-based champion\n"
                 f"{n} real SROIE receipts", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
