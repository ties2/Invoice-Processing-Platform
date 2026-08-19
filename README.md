# Invoice Processing Platform (IPP)

It's deliberately built as a layered platform where the model is a small,
swappable component and the value is the engineering around it: safe rollout of
new extractors, deterministic financial validation, confidence-based routing,
observability, and drift detection.

> make three specific
> production challenges concrete: legacy modernisation, safe A/B testing on live
> traffic, and the offline-vs-production corner-case gap.


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
                 validation, routing, pipeline
src/monitoring/  drift_detector
tests/           unit + API integration (SQLite, no external services)
```

## Run it

```bash
# Full stack (API + Postgres)
cp .env.example .env
docker compose up --build
# -> http://localhost:8000/docs

# Or locally against your own Postgres
pip install -r requirements.txt
uvicorn src.serving.app:app --reload

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

## Moving from simulation to real models (Path B — implemented)

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
