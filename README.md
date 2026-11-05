# Model-Aware Adaptive Retrieval-Augmented Generation

This repository is the workspace for research on **Model-Aware Adaptive Retrieval-Augmented Generation**. It preserves the original Adaptive-RAG implementation as a self-contained baseline while keeping our documentation, experiment code, results, and paper sources in a separate research tree.

## Directory structure

```text
Model-Aware-Adaptive-RAG/
├── adaptive-rag/   # Original Adaptive-RAG implementation
├── research/       # Our Model-Aware Adaptive-RAG work
├── README.md       # Workspace documentation
└── .gitignore      # Workspace-wide ignore policy
```

### `adaptive-rag/`

This directory contains the complete upstream implementation, including its original README, license, source code, configurations, scripts, prompts, evaluation tools, and distributed data archives. The implementation originated from [starsuzi/Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG). See [`adaptive-rag/LICENSE`](adaptive-rag/LICENSE) for licensing terms and [`adaptive-rag/README.md`](adaptive-rag/README.md) for the original setup and usage documentation.

The baseline files were relocated as a unit and should not be modified merely to support our research. New wrappers, analyses, and routing methods belong under `research/`.

### `research/`

This directory contains all project-owned work:

- `docs/` — Phase 0 codebase notes and the shared experiment plan.
- `paper/` — the two-column LaTeX results document, seven independently editable experiment tables, and figures.
- `experiments/` — research-owned drivers and analysis code by phase.
- `results/` — small, reviewable summaries and metadata by result family.

See [`research/README.md`](research/README.md) for the phase definitions and collaboration workflow.

## Running the original baseline after relocation

The upstream implementation contains many paths relative to its working directory. Always enter `adaptive-rag/` before following its original README:

```sh
cd adaptive-rag
```

Commands in the upstream documentation then remain unchanged. For example:

```sh
uvicorn serve:app --port 8000 --app-dir retriever_server
MODEL_NAME={model_name} uvicorn serve:app --port 8010 --app-dir llm_server
bash run_retrieval_dev.sh "$SYSTEM" "$MODEL" "$DATASET" "$LLM_PORT_NUM"
bash run_retrieval_test.sh "$SYSTEM" "$MODEL" "$DATASET" "$LLM_PORT_NUM"
python ./evaluate_final_acc.py
```

The baseline expects directories such as `base_configs/`, `prompts/`, `processed_data/`, `predictions/`, and `classifier/data/` relative to the current working directory. Running the scripts from the workspace root will generally fail unless every path is manually prefixed; changing into `adaptive-rag/` is the supported approach.

These examples document invocation only. They may require datasets, indices, model servers, dependencies, and credentials described by the upstream README.

## Git and upstream synchronization

This remains one Git repository rooted at `Model-Aware-Adaptive-RAG/`; `adaptive-rag/` is not a nested repository. The `origin` remote is the project repository, and `upstream` continues to identify the original Adaptive-RAG repository.

Because upstream still stores its files at repository root while this project stores them under `adaptive-rag/`, future upstream merges are no longer layout-identical. Fetch upstream normally, but review integrations on a dedicated branch and expect rename-aware conflict resolution or a subtree-style import. Never resolve such a merge by overwriting `research/`.

## Reproducibility boundary

Baseline reproduction should use the unmodified contents of `adaptive-rag/` and record the upstream/project commit, command, configuration, environment, model version, and dataset split. Research-specific code and outputs should be added only under `research/`. Large corpora, predictions, indices, checkpoints, generated LaTeX files, and credentials are intentionally excluded from version control.
