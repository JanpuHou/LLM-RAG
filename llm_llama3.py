# pip install transformers
# pip install accelerate

# git clone https://github.com/facebookresearch/llama
import ollama
response = ollama.chat(model=r'Llama3.2', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])

response = ollama.chat(model=r'Llama3.2', messages=[
  {
    'role': 'user',
    'content': '中華文化有什麽?',
  },
])
print(response['message']['content'])