"""Test Mivi with the official OpenAI Python SDK."""
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

print("=" * 60)
print("TEST 1: Simple Chat Completion")
print("=" * 60)
response = client.chat.completions.create(
    model="mivi",
    messages=[{"role": "user", "content": "Hello! Who are you?"}],
    temperature=0.2,
    max_tokens=64,
)
print(f"Model ID: {response.model}")
print(f"Response: {response.choices[0].message.content}")
print(f"Usage: {response.usage}\n")

print("=" * 60)
print("TEST 2: Streaming")
print("=" * 60)
stream = client.chat.completions.create(
    model="mivi",
    messages=[{"role": "user", "content": "Write a short haiku about coding."}],
    temperature=0.5,
    max_tokens=64,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n\n✅ OpenAI SDK test finished successfully!")
