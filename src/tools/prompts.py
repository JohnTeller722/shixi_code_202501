def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        return f.read()