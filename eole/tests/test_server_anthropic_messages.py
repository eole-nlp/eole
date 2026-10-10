"""Anthropic request conversion must handle Claude Code message layouts."""

import unittest

from eole.server.anthropic import AnthropicMessagesRequest, _anthropic_messages_to_openai


class TestServerAnthropicMessages(unittest.TestCase):
    def test_request_accepts_system_role_in_messages(self):
        request = AnthropicMessagesRequest(
            model="test",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Keep the answer short."},
            ],
        )

        self.assertEqual(request.messages[1].role, "system")

    def test_combines_top_level_and_message_system_content(self):
        request = AnthropicMessagesRequest(
            model="test",
            system=[{"type": "text", "text": "Top-level instructions."}],
            messages=[
                {"role": "user", "content": "First question"},
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Mid-conversation instructions.",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
            ],
        )

        converted = _anthropic_messages_to_openai(request.messages, request.system)

        self.assertEqual(
            converted,
            [
                {
                    "role": "system",
                    "content": "Top-level instructions.\nMid-conversation instructions.",
                },
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
            ],
        )

    def test_system_only_content_becomes_leading_message(self):
        converted = _anthropic_messages_to_openai(
            [
                {"role": "user", "content": "Question"},
                {"role": "system", "content": "Late instructions"},
            ]
        )

        self.assertEqual(
            converted,
            [
                {"role": "system", "content": "Late instructions"},
                {"role": "user", "content": "Question"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
