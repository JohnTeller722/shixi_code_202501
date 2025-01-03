from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1025/v1",
    api_key="sk-1234567890"
)

# comprehensive_1.txt
# detailed_1.txt
with open('data/test_data_QA_split/system_question/comprehensive_1.txt', 'r') as f:
    system = f.read()

with open('data/test_data_QA_split/user_examples/user.txt', 'r') as f:
    user = f.read()

response = client.chat.completions.create(
    model="Qwen2.5-72B",
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ],
    temperature=1.0,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
