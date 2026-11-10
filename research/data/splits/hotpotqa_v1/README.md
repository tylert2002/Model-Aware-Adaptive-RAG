# HotpotQA v1 candidate frozen splits

These manifests were generated deterministically with seed `20261109` from
`./processed_data/hotpotqa/test_subsampled.jsonl` inside
`adaptive-rag/processed_data.tar.gz`. They contain no IDs from the 500-question
Phase 0 development set.

| Split | Unique questions | Bridge | Comparison |
|---|---:|---:|---:|
| Calibration | 50 | 41 | 9 |
| Validation | 25 | 21 | 4 |
| Held-out test | 150 | 124 | 26 |

The top-level JSONL files preserve complete source records, including reference
answers and gold-context annotations, and are evaluation-only. Never place their
`answers_objects`, `contexts`, or `is_supporting` values in an LLM prompt.

The `inference/` files contain only dataset name, question ID, question text,
difficulty level, and question type. They are safe for assigning questions to
inference workers, but scoring must join predictions back to the evaluation-only
files by exact question ID.

Regenerate only in a clean output directory:

```sh
python3 research/experiments/splits/freeze_hotpotqa_v1.py
```

Verify the existing files without writing:

```sh
python3 research/experiments/splits/freeze_hotpotqa_v1.py --verify-existing
```

The generator refuses to overwrite existing outputs. These manifests remain
candidate-frozen pending a live retrieval/index compatibility check and user
approval. See `split_metadata.json` and the E0.0 report for provenance.
