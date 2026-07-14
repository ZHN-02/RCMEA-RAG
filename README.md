# RCMEA-RAG

RCMEA-RAG is a risk-controlled multimodal evidence-arbitration framework for
pancreas CT report generation. It separates image-derived pancreas-state
reasoning from report realization, with multimodal evidence used to guide the
final abnormal/normal state and structured pancreatic findings.

## Public Release

This repository contains a de-identified source implementation for the
training-side policy search, RCMEA-RAG gate and ROI-route analyses, matched
RAG baselines, and evidence-summary generation. The code is organized as five
portable stages:

- Step 01: training-side OOF policy search and route-evidence scoring.
- Step 02: adaptive ROI-route analysis.
- Step 03: matched plain-RAG and route-vote baselines.
- Step 04: gate evaluation and leave-one-route-out ablations.
- Step 05: evidence-summary generation.

It also provides a lightweight utility for extracting, normalizing, and
deduplicating pancreas report sections:

```python
from rcmea_rag.report_text_cleanup import clean_report_section

section = clean_report_section(raw_report_text)
```

The report-section utility depends only on the Python standard library.

## Installation

```bash
python -m pip install -e .
```

The research scripts require Python 3.10 or later and NumPy. Their input and
output roots are configured through `RCMEA_INPUT_ROOT` and `RCMEA_OUTPUT_ROOT`.

## Data and Models

The public repository intentionally excludes protected clinical data, report
text, retrieval inputs and embeddings, model weights, generated predictions,
and internal experimental artifacts.

## Citation

Citation information will be added with the final manuscript record.
