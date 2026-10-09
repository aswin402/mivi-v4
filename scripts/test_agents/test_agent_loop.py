"""Offline regression tests for the real example loop; only HTTP is replaced."""
import copy
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    "agent_loop", Path(__file__).with_name("02_agent_loop.py")
)
agent_loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_loop)


def tool_call(expression="15 * 8 + 30", call_id="call_1", name="calculator"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"expression": expression})},
    }


class AgentLoopTests(unittest.TestCase):
    def run_loop(self, calls, *, max_steps=2, terminal_reason="stop", final_content="finished"):
        requests = []
        responses = iter([
            {"choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": calls,
            }}]},
            {"choices": [{"finish_reason": terminal_reason, "message": {
                "role": "assistant", "content": final_content,
            }}]},
        ])

        def post(url, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            response = Mock()
            response.json.return_value = next(responses)
            return response

        with patch.object(agent_loop.requests, "post", side_effect=post), redirect_stdout(io.StringIO()):
            result = agent_loop.run_agent("offline regression", max_steps=max_steps)
        self.assertEqual(result, "finished")
        return requests

    def test_arithmetic_reply_carries_original_call_id(self):
        requests = self.run_loop([tool_call()])
        reply = requests[1]["json"]["messages"][-1]
        self.assertEqual(reply["content"], "150")
        self.assertEqual(reply.get("tool_call_id"), "call_1")

    def test_multiple_calls_have_one_correlated_reply_each(self):
        requests = self.run_loop([tool_call("2 + 3", "a"), tool_call("7 * 6", "b")])
        replies = requests[1]["json"]["messages"][-2:]
        self.assertEqual([(r.get("tool_call_id"), r["content"]) for r in replies],
                         [("a", "5"), ("b", "42")])

    def test_python_calls_are_rejected_without_execution(self):
        with patch("builtins.pow", return_value=8) as forbidden:
            requests = self.run_loop([tool_call('__import__("builtins").pow(2, 3)')])
        forbidden.assert_not_called()
        self.assertTrue(requests[1]["json"]["messages"][-1]["content"].startswith("Error:"))

    def test_unsupported_syntax_returns_tool_errors(self):
        for expression in ("True + 1", "'a' * 10", "[1][0]", "1 < 2", "2 ** 1000000"):
            with self.subTest(expression=expression):
                requests = self.run_loop([tool_call(expression)])
                self.assertTrue(requests[1]["json"]["messages"][-1]["content"].startswith("Error:"))

    def test_nonfinite_and_oversized_work_returns_tool_errors(self):
        for expression in ("1e309", "1e100 * 1e100", "9" * 100, "1+" * 300 + "1"):
            with self.subTest(expression=expression):
                requests = self.run_loop([tool_call(expression)])
                self.assertTrue(requests[1]["json"]["messages"][-1]["content"].startswith("Error:"))

    def test_arithmetic_failure_is_returned_to_model(self):
        requests = self.run_loop([tool_call("1 / 0")])
        self.assertTrue(requests[1]["json"]["messages"][-1]["content"].startswith("Error:"))

    def test_unknown_tool_is_returned_to_model(self):
        requests = self.run_loop([tool_call(name="unknown")])
        reply = requests[1]["json"]["messages"][-1]
        self.assertEqual(reply["role"], "tool")
        self.assertEqual(reply.get("tool_call_id"), "call_1")
        self.assertTrue(reply["content"].startswith("Error:"))

    def test_malformed_arguments_are_returned_to_model(self):
        for arguments in ("{", "[]", '{"expression":false}', '{"other":1}'):
            with self.subTest(arguments=arguments):
                call = tool_call()
                call["function"]["arguments"] = arguments
                requests = self.run_loop([call])
                self.assertTrue(requests[1]["json"]["messages"][-1]["content"].startswith("Error:"))

    def test_missing_or_duplicate_ids_fail_before_tool_execution(self):
        for calls in ([tool_call(call_id="")], [tool_call(call_id=None)],
                      [tool_call(call_id="same"), tool_call(call_id="same")]):
            with self.subTest(calls=calls), self.assertRaisesRegex(ValueError, "tool.*ID"):
                self.run_loop(calls)

    def test_http_requests_have_finite_timeout(self):
        requests = self.run_loop([tool_call()])
        for request in requests:
            timeout = request.get("timeout")
            self.assertIsNotNone(timeout)
            values = timeout if isinstance(timeout, tuple) else (timeout,)
            self.assertTrue(all(0 < value <= 300 for value in values))

    def test_supported_arithmetic(self):
        for expression, expected in (
            (" -(2 + 3) * +4 ", "-20"), ("7 / 2", "3.5"),
            ("7 // 2", "3"), ("7 % 2", "1"), ("2.5 + 0.5", "3.0"),
        ):
            with self.subTest(expression=expression):
                requests = self.run_loop([tool_call(expression)])
                self.assertEqual(requests[1]["json"]["messages"][-1]["content"], expected)

    def test_bounded_depth_and_nodes(self):
        for expression in ("-" * 20 + "1", "+".join(["1"] * 45)):
            with self.subTest(expression=expression):
                requests = self.run_loop([tool_call(expression)])
                self.assertTrue(requests[1]["json"]["messages"][-1]["content"].startswith("Error:"))

    def test_invalid_tool_envelopes_return_correlated_errors(self):
        for change in ({"type": "unknown"}, {"function": None}, {"function": {}}):
            with self.subTest(change=change):
                call = tool_call()
                call.update(change)
                requests = self.run_loop([call])
                reply = requests[1]["json"]["messages"][-1]
                self.assertEqual(reply["tool_call_id"], "call_1")
                self.assertTrue(reply["content"].startswith("Error:"))

    def test_empty_tool_calls_are_not_reported_as_final_answer(self):
        with self.assertRaisesRegex(ValueError, "tool calls"):
            self.run_loop([])

    def test_step_exhaustion_is_not_success(self):
        with self.assertRaisesRegex(RuntimeError, "step limit"):
            self.run_loop([tool_call()], max_steps=1)

    def test_invalid_step_budget_fails_before_http(self):
        for budget in (0, -1, True, 1.5):
            with self.subTest(budget=budget), patch.object(agent_loop.requests, "post") as post:
                with self.assertRaisesRegex(ValueError, "max_steps"):
                    agent_loop.run_agent("unused", max_steps=budget)
                post.assert_not_called()

    def test_truncated_or_unknown_completion_is_not_success(self):
        for reason in ("length", "content_filter", "unknown", None):
            with self.subTest(reason=reason), self.assertRaisesRegex(RuntimeError, "finish reason"):
                self.run_loop([tool_call()], terminal_reason=reason)

    def test_empty_or_nontext_final_answer_is_not_success(self):
        for content in (None, "", "   ", {"text": "finished"}):
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "final answer"):
                self.run_loop([tool_call()], final_content=content)


if __name__ == "__main__":
    unittest.main()
