from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# Completion API
stream=True
prompt="Hello, my name is"
completion = client.completions.create(
    model=model,
    prompt=prompt,
    stream=stream,
    )

print("Prompt:", prompt)
print("Completion results:")
if stream:
    for c in completion:
        print(c.choices[0].text, end="")
    print()
else:
    print(completion)
