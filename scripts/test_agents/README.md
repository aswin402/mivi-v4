# Agent examples

These scripts are examples, not a comprehensive provider-conformance or model-quality suite.
The OpenAI-compatible chat endpoint returns tool calls for the client to execute;
`03_native_agent.py` instead exercises Mivi's server-side agent endpoint.

## Calculator loop

With Mivi serving locally on port 8080 and Python's `requests` installed:

```sh
python3 scripts/test_agents/02_agent_loop.py
```

The example uses the `mivi` model alias and no API key on loopback. Its calculator
interprets only real numeric literals, parentheses, unary `+`/`-`, and binary
`+`, `-`, `*`, `/`, `//`, `%`. It never evaluates model text as Python code.
Names, calls, attributes, containers, booleans, complex numbers, and powers are
rejected. Limits are 512 expression characters, 128 AST nodes, depth 16,
256-bit integers, finite floats of magnitude at most `1e100`, and 4,096 characters
for JSON-encoded tool arguments. Limits apply to intermediate arithmetic too.

Each tool reply carries the original `tool_call_id`. Invalid arguments and
unknown tools produce error replies; missing or duplicate IDs abort the loop
before tool execution. HTTP errors, timeouts, incomplete finish reasons, and
step exhaustion are failures rather than successful final answers.

Requests use a 5-second connect timeout and a 120-second read timeout. These are
socket-operation limits, **not a task-wide deadline** or proof of server-side
cancellation. The example permits five model calls by default.

Run the focused offline regressions without loading a model or contacting a server:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/test_agents -p test_agent_loop.py -v
```

The tests run the actual arithmetic/tool loop with only the HTTP transport
replaced. They do not establish live agent success, inference performance, or
semantic correctness of the model's final answer.

## Ideas, inspirations, and sources

- Replace the unsafe execution found in Mivi's own example with an explicit
  allowlist interpreter using [Python AST nodes](https://docs.python.org/3/library/ast.html)
  and [numeric operators](https://docs.python.org/3/library/operator.html).
  No external implementation was copied.
- Distinguish HTTP/protocol completion from answer correctness, following
  [Colibri's benchmark protocol](https://github.com/JustVugg/colibri/blob/main/docs/benchmarking.md).
- Timeout boundaries follow the [Requests documentation](https://requests.readthedocs.io/en/latest/user/quickstart/#timeouts).
