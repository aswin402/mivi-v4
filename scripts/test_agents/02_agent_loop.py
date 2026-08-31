"""Test Mivi tool calling in a simulated agent loop with calculator and directory listing."""
import json
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

def run_agent(task: str, max_steps: int = 5):
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
        )
        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        print(f"  Content: {msg.get('content', '(none)')}")
        print(f"  Finish reason: {finish_reason}")

        if finish_reason == "tool_calls" and msg.get("tool_calls"):
            messages.append(msg)
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                tool_name = fn["name"]
                args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                print(f"  🔧 Tool call: {tool_name}({args})")
                if tool_name == "calculator":
                    result = str(eval(args["expression"]))
                    print(f"  📋 Result: {result}")
                    messages.append({"role": "tool", "name": tool_name, "content": result})
        else:
            print(f"\n✅ FINAL ANSWER: {msg.get('content', '(empty)')}")
            return msg.get("content", "")

if __name__ == "__main__":
    run_agent("What is 15 * 8 + 30?")
