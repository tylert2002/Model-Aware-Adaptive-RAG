import json
import os
import unittest
from unittest.mock import patch

try:
    from .nrp_adapter import (
        NRPChatAdapter,
        NRPConfigurationError,
        NRPConfig,
        NRPRequestError,
    )
except ImportError:
    from nrp_adapter import (
        NRPChatAdapter,
        NRPConfigurationError,
        NRPConfig,
        NRPRequestError,
    )


class NRPAdapterTests(unittest.TestCase):
    def setUp(self):
        self.config = NRPConfig(
            api_key="test-placeholder",
            max_retries=2,
            backoff_initial_seconds=0,
            backoff_max_seconds=0,
        )

    def test_parses_openai_compatible_response_and_usage(self):
        payload = {
            "id": "request-1",
            "model": "gemma-small",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        }

        def transport(request, timeout):
            self.assertNotIn("test-placeholder", request.full_url)
            return 200, {}, json.dumps(payload).encode()

        result = NRPChatAdapter(self.config, transport=transport).generate("hello", model="gemma-small")
        self.assertEqual(result.text, "ok")
        self.assertEqual(result.input_tokens, 4)
        self.assertEqual(result.output_tokens, 1)
        self.assertEqual(result.attempts, 1)

    def test_retries_transient_status_with_bound(self):
        attempts = []

        def transport(request, timeout):
            attempts.append(1)
            if len(attempts) < 3:
                return 429, {}, b"{}"
            return 200, {}, json.dumps(
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
            ).encode()

        result = NRPChatAdapter(self.config, transport=transport, sleep=lambda _: None).generate(
            "hello", model="gemma"
        )
        self.assertEqual(result.attempts, 3)

    def test_does_not_retry_invalid_response(self):
        adapter = NRPChatAdapter(self.config, transport=lambda request, timeout: (200, {}, b"{}"))
        with self.assertRaises(NRPRequestError) as context:
            adapter.generate("hello", model="gemma")
        self.assertEqual(context.exception.category, "invalid_response")

    def test_preserves_sanitized_http_diagnostic_without_secret(self):
        secret = "sensitive-test-key"
        config = NRPConfig(api_key=secret, max_retries=0)
        body = json.dumps({"error": {"message": f"model forbidden for {secret}"}}).encode()
        adapter = NRPChatAdapter(config, transport=lambda request, timeout: (403, {}, body))
        with self.assertRaises(NRPRequestError) as context:
            adapter.generate("hello", model="gemma-small")
        self.assertEqual(context.exception.status_code, 403)
        self.assertEqual(context.exception.diagnostic, "model forbidden for [REDACTED]")
        self.assertNotIn(secret, str(context.exception.diagnostic))

    def test_rejects_unlisted_model_instead_of_falling_back(self):
        adapter = NRPChatAdapter(self.config, transport=lambda request, timeout: (200, {}, b"{}"))
        with self.assertRaises(NRPConfigurationError):
            adapter.generate("hello", model="service")

    def test_environment_requires_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(NRPConfigurationError):
                NRPConfig.from_env()


if __name__ == "__main__":
    unittest.main()
