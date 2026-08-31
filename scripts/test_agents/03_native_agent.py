"""Test Mivi native autonomous agent endpoint (/v1/mivi/agent) via SSE streaming."""
import json
import requests

def run_native_agent(task: str, max_steps: int = 5):
    print(f"\n{'='*60}\nNATIVE AGENT TASK: {task}\n{'='*60}\n")
    resp = requests.post(
        "http://127.0.0.1:8080/v1/mivi/agent",
        json={"task": task, "max_steps": max_steps},
        stream=True,
        headers={"Accept": "text/event-stream"},
    )

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                print("\n\n✅ Native agent completed task!")
                break
            try:
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if delta.get("thinking"):
                    print(f"  💭 {delta['thinking']}", end="")
                if delta.get("content"):
                    print(delta["content"], end="", flush=True)
            except json.JSONDecodeError:
                print(f"  [raw] {data}")

if __name__ == "__main__":
    run_native_agent("List the files in the current workspace directory.")
