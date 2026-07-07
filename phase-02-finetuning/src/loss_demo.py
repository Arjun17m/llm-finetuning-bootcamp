import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": "What is Artificial Intelligence?"
    },
    {
        "role": "assistant",
        "content": "Artificial Intelligence is the field of building machines that can perform tasks requiring human intelligence."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

encoded = tokenizer(text, return_tensors="pt").to(model.device)
print("=" * 80)
print("Encoded Input IDs")
print("=" * 80)
print(encoded)
labels = encoded["input_ids"].clone()

with torch.no_grad():
    outputs = model(
        **encoded,
        labels=labels
    )

print("=" * 80)
print("LOSS")
print("=" * 80)
print(outputs.loss)

print("\n")

print("=" * 80)
print("LOGITS SHAPE")
print("=" * 80)
print(outputs.logits.shape)