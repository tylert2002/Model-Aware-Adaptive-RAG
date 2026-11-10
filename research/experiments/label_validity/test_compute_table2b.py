from __future__ import annotations

import json
import unittest

from research.experiments.label_validity.compute_table2b import (
    OUTPUT,
    assert_same_question_ids,
    compute_row,
)


class Table2bConsistencyTests(unittest.TestCase):
    def test_rejects_different_metric_question_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "metric question-set mismatch"):
            assert_same_question_ids({"auto": {"q1", "q2"}, "oracle": {"q1"}})

    def test_row_uses_one_set_and_regret_is_difference(self) -> None:
        indexed = {}
        values = {"q1": {"A": (0.0, 1), "B": (0.5, 2), "C": (0.5, 3)}, "q2": {"A": (0.0, 1), "B": (0.0, 2), "C": (0.0, 3)}}
        for qid, strategies in values.items():
            for strategy, (f1, tokens) in strategies.items():
                indexed[(qid, strategy)] = {"f1": f1, "total_tokens": tokens}
        row = compute_row(indexed, {"q1": "C", "q2": "A"})
        self.assertEqual(row["n"], 2)
        self.assertAlmostEqual(row["oracle_mean_f1"], 0.25)
        self.assertAlmostEqual(row["auto_label_mean_f1"], 0.25)
        self.assertAlmostEqual(row["mean_f1_regret"], 0.0)
        self.assertEqual(len(set(row["metric_question_id_sha256"].values())), 1)

    def test_generated_rows_record_identical_metric_sets(self) -> None:
        payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            self.assertEqual(len(set(row["metric_question_id_sha256"].values())), 1)
            self.assertGreaterEqual(row["oracle_mean_f1"] + 1e-12, row["auto_label_mean_f1"])
            self.assertAlmostEqual(
                row["mean_f1_regret"], row["oracle_mean_f1"] - row["auto_label_mean_f1"], places=12
            )


if __name__ == "__main__":
    unittest.main()
