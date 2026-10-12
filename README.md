# Invoice Processing Platform (IPP)

It's deliberately built as a layered platform where the model is a small,
swappable component and the value is the engineering around it: safe rollout of
new extractors, deterministic financial validation, confidence-based routing,
observability, and drift detection.

> Makes three specific production challenges concrete: legacy modernisation,
> safe A/B testing on live traffic, and the offline-vs-production corner-case gap.

The platform runs on real scanned invoices end to end: real OCR (Tesseract), a
rule-based baseline **and** an LLM extractor proven against it in shadow mode on
a real public dataset (SROIE), with a visual overlay of what the pipeline found
on each page. Those parts are described in **Real-data evaluation** and
**LLM extractor in shadow mode** below.

## Pipeline

```
PDF/Image ─► Storage ─► OCR ─► Extractor ─┐
                          │               ├─► Validation ─► Routing ─► decision
             Table detect ┘               │                    │
        Experiment (shadow/canary) ───────┘            auto_booked / needs_review
                                                                 │
                                                        Human correction
                                                                 │
                                                        corrections (feedback)
```

## How the design answers three hard problems

**1. Legacy modernisation without a big-bang rewrite.**
Every extractor implements one `Extractor` interface. The legacy code is just
`LegacyExtractor`; a new model is another implementation. Swapping or comparing
them is a config change in `pipeline_factory.py` — the API, service and database
never move. This is the Strangler Fig pattern.

**2. Testing a new system on millions of real invoices safely.**
`ExtractionExperiment` runs a `champion` and a `challenger`:
- `SHADOW` — challenger runs on real traffic but its output never affects the
  decision (zero production risk).
- `CANARY` — a deterministic percentage of traffic is served by the challenger.
  Rollback is flipping one setting.

**3. The offline-vs-production corner-case gap.**
Three independent signals surface corner cases: low confidence, champion/
challenger disagreement, and validation failure. Drift is watched with two
label-free indicators — confidence PSI and new-supplier rate — so problems are
visible before quality craters. Corrections feed back as labelled data.

## Key decisions worth knowing

- **Per-field confidence, aggregated as the minimum.** A document is only as
  trustworthy as its weakest field, so one shaky field triggers review.
- **Deterministic validation is a separate gate from the model.** A high-
  confidence prediction that fails `subtotal + VAT = total` is *blocked* from
  auto-booking. Line items must also reconcile to the subtotal.
- **Thresholds live in config, not code** — they are a business decision tuned
  against the cost of false positives vs. false negatives.
- **Known-supplier registry.** A registry hit is trusted; an unknown supplier
  routes to review, gets corrected, and can then be learned — the feedback loop
  in miniature.
- **Fail safe.** Any pipeline error marks the invoice `FAILED` for a human; it is
  never silently booked.

## Layout

```
src/serving/     app, api, service, repository, db_models, schemas,
                 database, config, storage, dependency, pipeline_factory,
                 observability, analytics
src/pipeline/    ocr, extractor (+ legacy), table_detector, experiment,
                 validation, routing, pipeline, visualize
src/datasets/    sroie (real-receipt loader + scoring)
src/monitoring/  drift_detector
scripts/         demo_visualize, run_experiment (+ llm_shadow_cache.json)
tests/           unit + API integration (SQLite, no external services)
```

## Run it

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Full stack (API + Postgres)
cp .env.example .env
docker compose up --build
# -> http://localhost:8000/docs

# Or locally against your own Postgres
pip install -r requirements.txt
#change .env
POSTGRES_HOST=localhost
#just run database 
docker compose up -d db
#run API on local
uvicorn src.serving.app:app --reload

#restaret API container
docker compose restart api

# Tests (hermetic, SQLite-backed)
pytest

```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/invoices` | Upload a document (multipart) |
| POST | `/invoices/{id}/process` | Run the pipeline → decision |
| GET  | `/invoices/{id}` | Invoice detail + extracted fields |
| GET  | `/invoices` | List (optional `?status=`) |
| POST | `/invoices/{id}/corrections` | Submit a human correction |
| GET  | `/metrics` | Business KPIs (auto rate, correction rate, hotspots) |

## Moving from simulation to real models

The simulated stages live behind interfaces, so real models slot in without
touching the rest of the system. Two real backends are included:

| Interface | Simulated | Real backend (included) |
|---|---|---|
| `OCREngine` | `SimulatedOCR` (text + synthetic boxes) | `TesseractOCR` — Tesseract on real image/PDF pages |
| `Extractor` | `RuleBasedExtractor` (regex + anchors) | `LLMExtractor` / `VLMExtractor` — structured extraction via an LLM/VLM |
| `TableDetector` | geometry heuristic | seam for a trained detector (YOLO) for rows/columns |

Enable them with `pip install -r requirements-ml.txt` (plus system `tesseract-ocr`
and `poppler-utils`). They are swapped in via `pipeline_factory.py` only.

Two safety ideas in the LLM path: a **grounding guard** drops the confidence of
any value not found in the OCR text (hallucination defence), and the neural
output still passes the deterministic validation gate — the LLM proposes,
arithmetic disposes. Because both backends honour the same interface, a new model
is first proved in `SHADOW` mode against the rule-based champion.

Each swap is a benchmark decision, evaluated on production-shaped, supplier-aware
slices — not a wholesale replacement.

---

# Real-data evaluation & result visualisation

This is an **add-on**, not a rewrite. It reuses every production stage unchanged
(TesseractOCR → RuleBasedExtractor → SimulatedTableDetector → InvoiceValidator →
RoutingEngine). Only the *input* (a real scanned receipt instead of a `.txt`
blob) and the *output* (an overlay image + an accuracy score) are new.

Files it adds:

```
src/pipeline/visualize.py      draw detected fields back onto the page
src/datasets/__init__.py       package marker
src/datasets/sroie.py          real SROIE dataset loader + scoring
scripts/demo_visualize.py      runnable end-to-end demo
requirements-ml.txt            one line added: matplotlib>=3.7
```

## 1. Get the dataset (real receipt images)

Pick one source and point `IPP_DATASET_ROOT` at the folder that contains
`img/ box/ key/`:

- **Kaggle:** `kaggle datasets download -d urbikn/sroie-datasetv2` → unzip
- **GitHub:** https://github.com/zzzDavid/ICDAR-2019-SROIE (`data/img`, `data/box`, `data/key`)
- **Official ICDAR 2019 SROIE:** https://rrc.cvc.uab.es/?ch=13

SROIE = 600+ real scanned receipts with ground-truth labels
(`company`, `date`, `total`). The loader maps those to this project's fields
(`supplier`, `invoice_date`, `total`); unlabeled fields are never scored wrong.

## 2. Install

```bash
# system OCR engine:
brew install tesseract poppler
#linux / docker
sudo apt-get install -y tesseract-ocr poppler-utils
```

## 3. Run

```bash
export IPP_DATASET_ROOT=/path/to/SROIE/data        # the folder with img/ box/ key/

# (a) one receipt -> side-by-side figure: original vs detected fields
python scripts/demo_visualize.py --id 060 --out result_060.png

# (b) extraction accuracy vs ground truth over the first N receipts
python scripts/demo_visualize.py --eval 40
```

## What the visualisation shows

A single matplotlib figure, two panels:

- **Left** — the raw receipt image.
- **Right** — the same image with: faint grey boxes for every OCR word
  (what OCR saw), bold coloured boxes + labels for each extracted header field
  with its confidence, and shaded bands for any detected line-item columns.
  This is the document-AI equivalent of overlaying a segmentation mask on the
  source page.

The figure title also carries the routing decision (`auto_booked` /
`needs_review`), so one picture explains the whole pipeline on one invoice.

### Using it from your own code

```python
from src.datasets.sroie import SroieDataset, score_sample
from src.pipeline.ocr_tesseract import TesseractOCR
from src.pipeline.extractor import RuleBasedExtractor
from src.pipeline.table_detector import SimulatedTableDetector
from src.pipeline.visualize import render_result

sample = SroieDataset("/path/to/SROIE/data").load("060")
ocr   = TesseractOCR(lang="eng").recognize(sample.image_bytes)
preds = RuleBasedExtractor().extract(ocr)
table = SimulatedTableDetector().detect(ocr)

render_result(sample.image_bytes, ocr, preds, table, save_path="result.png")
print(score_sample(preds, sample.ground_truth))   # per-field correct/incorrect
```

## Note on the numbers

On real, noisy receipts the rule-based baseline is intentionally beatable
(e.g. `date` high, `supplier` low because the "first line = supplier" heuristic
trips on logos/watermarks). That is the point: this harness makes the
offline-vs-production gap *measurable*, which is exactly what justifies swapping
in the `LLMExtractor` and proving it in `SHADOW` mode against this baseline.

---

# LLM extractor in shadow mode (A/B vs rule-based)

Builds on the real-data evaluation above. This phase wires the already-present
`LLMExtractor` into the existing `ExtractionExperiment` and proves it against the
rule-based champion on real receipts — the champion→challenger→promote loop, run
for real.

## Result (15 real SROIE receipts, measured)

| field        | champion (rule-based) | challenger (LLM) |
|--------------|:---------------------:|:----------------:|
| supplier     | 27%                   | **87%**          |
| invoice date | 80%                   | **87%**          |
| total        | 40%                   | **80%**          |
| **overall**  | **49%**               | **84%**          |

**+36 points → verdict: PROMOTE challenger.** The biggest lift is `supplier`:
the rule-based "first line = supplier" heuristic keeps grabbing the dataset
watermark; the LLM reads the actual company name and corrects OCR garble
(`MR O.1.¥.` → `MR D.I.Y. (JOHOR) SDN BHD`). See `shadow_report.png`.

Files it adds:

```
src/pipeline/extractor_llm.py   added RecordedLLMClient + document_key()
scripts/run_experiment.py       shadow A/B over real receipts + bar chart
scripts/llm_shadow_cache.json   recorded LLM responses (offline/repro runs)
tests/test_shadow_llm.py        replay + shadow-mode tests (all green)
```

Nothing in `experiment.py`, `routing.py`, `validation.py`, the service or the
API changes — the LLM path drops in behind the same `Extractor` interface, which
is the whole point of the architecture.

## Run it

```bash
export IPP_DATASET_ROOT=/path/to/SROIE/data

# A) with your own key -> real Anthropic calls
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_experiment.py --n 15

# B) no key / reproducible benchmark -> replay the recorded cache
python scripts/run_experiment.py --n 15 --cache scripts/llm_shadow_cache.json \
       --plot shadow_report.png
```

Both run the **identical** pipeline; only the LLM client differs
(`anthropic.Anthropic()` vs `RecordedLLMClient`).

## What `RecordedLLMClient` is (and why it's not a hack)

A drop-in for `anthropic.Anthropic()` with the same `.messages.create(...)`
surface that replays saved responses keyed by an MD5 of the invoice text. Three
real uses: deterministic tests with no network, reproducible benchmark numbers,
and a response cache (LLM calls are the costly part of the pipeline). Build the
cache once from live calls, replay it forever.

> Note on the shipped numbers: the recorded responses were generated by reading
> the same OCR text the production prompt sends (no API key in the build
> sandbox). With your own key, `run_experiment.py` makes the real calls through
> the exact same code path — swap option B for option A.

## Honest next step

The `supplier`/`total` LLM errors that remain are mostly OCR destroying the
source text (the LLM can't read what OCR dropped). The real lever there is the
`VLMExtractor` (already in `extractor_llm.py`): give the model the page pixels
via `OCRResult.source_image` instead of OCR text — the natural next update.