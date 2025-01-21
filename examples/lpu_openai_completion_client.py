import sys

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
stream = True
prompt = "Act like an experienced HR Manager. Develop a human resources strategy for retaining top talents in a competitive industry. Industry: (e.g Energy( Workforce: (e,g 550) Style: (e.g Formal) Tone: (e.g Convincing)"
completion = client.completions.create(
    model=model,
    prompt=prompt,
    frequency_penalty=1.2,
    max_tokens=40,
    temperature=1.0,
    top_p=0.8,
    stream=stream,
)

print("Prompt:", prompt)
print("Completion results:")
if stream:
    # print streaming output as a string
    result = ""
    for c in completion:
        chunk = c.choices[0].text
        result += chunk
        # if you want to see the progress, print to stderr
        print(chunk, end="", file=sys.stderr, flush=True)
    print("\n",file=sys.stderr)
    # print the final result to stdout
    print(result)
else:
    print(completion)
