# Model-Aware Adaptive-RAG Research Workspace

This directory contains research-owned documentation, experiment code, results, and LaTeX sources. The original Adaptive-RAG implementation remains separately preserved under `../adaptive-rag/`, including its license, attribution, scripts, configurations, and baseline code.

## Directory guide

- `config/`: shared model, dataset, and experiment configurations added incrementally.
- `src/`: future shared research code.
- `docs/`: the experiment roadmap and Phase 0 implementation notes.
- `paper/`: the experimental tables collection; update individual tables in `paper/tables/`.
- `experiments/`: one directory per planned experiment, with the existing NRP pilot adapter kept in `experiments/pilot/`.
- `results/`: small, reviewable outputs grouped by experiment. Large artifacts remain outside Git.

## Experiment index

| Experiment | Related Table | Implementation Status |
|---|---|---|
| pilot | P1 | API Adapter Ready; Pipeline Pending |
| label_validity | T2 | Not Implemented |
| retrieval_gain | T3 | Not Implemented |
| benchmark | T4 | Not Implemented |
| cross_model | T5 | Not Implemented |
| ablation | T6 | Not Implemented |

The formal sequence begins with pilot feasibility, followed by label validation, model-specific retrieval-gain profiling, the main benchmark, cross-model adaptation, and ablation. The detailed roadmap is in `docs/EXPERIMENT_PLAN.md`; no unimplemented experiment directory should be treated as runnable.

To reproduce the baseline without modifying it, change into `../adaptive-rag/` and follow that directory's original README and commands from there. Keep wrappers, new routing behavior, and experiment outputs under `research/`.
