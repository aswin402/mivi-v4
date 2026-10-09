"""Test Mivi tool calling with a bounded, arithmetic-only calculator.

Model-generated expressions are untrusted: never execute them as Python code.
"""
import ast
import json
import math
import operator
import requests

BASE_URL = "http://127.0.0.1:8080/v1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression and return the numerical result",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression like '25 * 4 + 10'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def calculate(expression):
    """Interpret numeric arithmetic only, with bounded input and intermediate values."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 512:
        raise ValueError("expression must be a nonempty string of at most 512 characters")
    tree = ast.parse(expression.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("expression is too complex")
    binary = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    }
    unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def checked(value):
        if type(value) is int:
            if value.bit_length() > 256:
                raise ValueError("integer exceeds 256 bits")
        elif type(value) is float:
            if not math.isfinite(value) or abs(value) > 1e100:
                raise ValueError("number is nonfinite or too large")
        else:
            raise ValueError("only real numeric literals are allowed")
        return value

    def visit(node, depth=0):
        if depth > 16:
            raise ValueError("expression is too deeply nested")
        if isinstance(node, ast.Constant):
            return checked(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return checked(binary[type(node.op)](visit(node.left, depth + 1),
                                                 visit(node.right, depth + 1)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            return checked(unary[type(node.op)](visit(node.operand, depth + 1)))
        raise ValueError("only +, -, *, /, //, %, parentheses and numeric literals are allowed")

    return visit(tree.body)


def execute_tool(call):
    """Return a correlated result even when a valid tool call has invalid arguments."""
    function = call.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    try:
        if call.get("type") != "function" or name != "calculator":
            raise ValueError("unsupported tool; only calculator is available")
        args = function.get("arguments")
        if isinstance(args, str):
            if len(args) > 4096:
                raise ValueError("tool arguments are too large")
            args = json.loads(args)
        if not isinstance(args, dict) or set(args) != {"expression"}:
            raise ValueError("calculator requires exactly one expression argument")
        result = str(calculate(args["expression"]))
    except (ValueError, SyntaxError, ArithmeticError, RecursionError) as error:
        result = f"Error: {error}"
    return {"role": "tool", "tool_call_id": call["id"], "content": result}


def run_agent(task: str, max_steps: int = 5):
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    print(f"\n{'='*60}\nAGENT TASK: {task}\n{'='*60}")
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to tools. Use tools when needed to answer questions accurately.",
        },
        {"role": "user", "content": task},
    ]

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "mivi",
                "messages": messages,
                "tools": TOOLS,
                "temperature": 0.1,
                "max_tokens": 128,
            },
            timeout=(5, 120),
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason")

        print(f"  Content: {msg.get('content', '(none)')}")
        print(f"  Finish reason: {finish_reason}")

        if finish_reason == "tool_calls":
            calls = msg.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                raise ValueError("tool calls must be a list with unique nonempty tool call IDs")
            ids = [call.get("id") if isinstance(call, dict) else None for call in calls]
            if any(not isinstance(call_id, str) or not call_id.strip() for call_id in ids):
                raise ValueError("tool call IDs must be nonempty strings")
            if len(set(ids)) != len(ids):
                raise ValueError("tool call IDs must be unique")
            messages.append(msg)
            for call in calls:
                reply = execute_tool(call)
                print(f"  📋 Result: {reply['content']}")
                messages.append(reply)
        elif finish_reason == "stop":
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("final answer must be nonempty text")
            print(f"\n✅ FINAL ANSWER: {content}")
            return content
        else:
            raise RuntimeError(f"unsupported or incomplete finish reason: {finish_reason!r}")

    raise RuntimeError("agent step limit reached without a final answer")

if __name__ == "__main__":
    run_agent("What is 15 * 8 + 30?")
