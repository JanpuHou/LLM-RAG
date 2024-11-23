# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="MediaTek-Research/Breexe-8x7B-Instruct-v0_1")
pipe(messages)