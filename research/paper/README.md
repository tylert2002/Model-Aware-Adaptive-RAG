# Experimental Results and Progress Tracker

This directory contains the living experimental record for **Model-Aware Adaptive Retrieval-Augmented Generation**. It is designed both as a publication-quality results shell and as a shared roadmap. It contains no claimed experimental measurements: every unmeasured result is represented by `--` in the rendered document.

## Research phases

1. **Phase 0 — Reproduction and label validity:** reproduce the original evaluation, run strategies A/B/C on identical questions, construct empirical oracles, and analyze agreement, ties, and regret.
2. **Phase 1 — Model profiling:** measure retrieval gains separately for LLM 1 and LLM 2.
3. **Phase 2 — Method construction:** build the Retrieval Gain Matrix and the model-aware router.
4. **Phase 3 — Main evaluation:** compare fixed baselines, the original adaptive router, and the proposed method on quality and efficiency.
5. **Phase 4 — Analysis:** measure calibration sample efficiency and perform component ablations.

## Tables

- **Table 0:** team progress tracker, dependencies, outputs, owners, and status.
- **Table 1:** supported datasets, split sizes, retrieval corpora, and evaluation metrics.
- **Table 2:** agreement and regret of original automatic labels relative to empirical quality oracles.
- **Table 3:** model-specific gains from single-step and multi-step retrieval.
- **Table 4:** main answer-quality and retrieval-efficiency comparison.
- **Table 5:** transfer of router profiles across LLMs and model-specific adaptation.
- **Table 6:** ablations of question features, model profiles, and cost-aware selection.

The separate configuration table in `main.tex` records model versions, generation settings, retrieval settings, serving conditions, and the evaluation-code revision.

## Compile

From this directory, run:

```sh
latexmk -pdf main.tex
```

If `latexmk` is unavailable, run `pdflatex main.tex` twice. The source uses only standard LaTeX packages and no conference-specific style file.

## Updating results

Each numbered table has one source file under `tables/`. Update only the relevant file, preserving table labels, captions, method names, column names, and experiment IDs. Replace `\NA` with a measured value only after the associated artifacts and protocol are archived. Do not estimate missing values or copy values across models or datasets.

When adding or updating a result:

1. Freeze and record the data split and question identifiers.
2. Record model, prompt, generation, retriever, index, and code versions.
3. Keep calibration, validation, and evaluation questions disjoint.
4. Store per-question predictions, F1/EM, routing choices, retrieval calls, and relevant timing/cost data.
5. State the cost scope and price date; retrieval, inference, and total API cost are different quantities.
6. Compile the document and inspect the PDF for overflow and consistency.
7. Update Table 0's status and owner.

Boldface may be added for verified best results after a complete, comparable block is available. Never bold placeholders.

## Status values

- **Not Started:** no qualifying experiment run has begun.
- **In Progress:** work is active, but the required output is incomplete or unverified.
- **Blocked:** progress depends on a documented unresolved prerequisite.
- **Complete:** the required output, configuration, and supporting artifacts have been produced and checked.

All experiments begin as **Not Started**. A status describes completion, not whether the research hypothesis was supported.
