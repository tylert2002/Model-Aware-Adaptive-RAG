from __future__ import annotations

import unittest

from .offline_validation import DeterministicMockAdapter


class OfflineValidationTests(unittest.TestCase):
    def test_mock_call_order_and_no_network_configuration(self):
        adapter = DeterministicMockAdapter("fixture answer")
        self.assertEqual(adapter.config.api_base, "offline://no-network")
        outputs = [adapter.generate(f"prompt {index}", model="offline-mock").text for index in range(5)]
        adapter.assert_complete()
        self.assertIn("related entity", outputs[2])
        self.assertIn("fixture answer", outputs[-1])
        self.assertEqual(
            [call["expected_stage"] for call in adapter.calls],
            ["A:final-answer", "B:final-answer", "C:reasoning-step-1", "C:reasoning-step-2", "C:final-answer"],
        )

    def test_mock_refuses_real_model_alias(self):
        adapter = DeterministicMockAdapter("fixture answer")
        with self.assertRaises(RuntimeError):
            adapter.generate("prompt", model="gemma-small")


if __name__ == "__main__":
    unittest.main()
