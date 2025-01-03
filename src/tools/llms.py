from openai import OpenAI

class LLM:
    def __init__(self, base_url, api_key):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def llm_chat(self, model, messages, temperature=1.0):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False
        )
        return response.choices[0].message.content

