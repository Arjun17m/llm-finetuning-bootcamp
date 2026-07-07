from datasets import load_dataset
import torch
from transformers import AutoTokenizer

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = load_dataset(
    "json",
    data_files="../data/instruction_dataset.json"
)

example = dataset["train"][0]

messages = [
    {
        "role": "user",
        "content": f"{example['instruction']}\n\n{example['input']}"
    },
    {
        "role": "assistant",
        "content": example["output"]
    }
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

encoded = tokenizer(
    formatted_prompt,
    return_tensors="pt"
)


labels = encoded["input_ids"].clone()

print("=" * 80)
print("LABELS")
print("=" * 80)
print(labels)

print()

print("INPUT IDS == LABELS ?")

print(torch.equal(encoded["input_ids"], labels))

print("=" * 80)
print("INPUT IDS")
print("=" * 80)
print(encoded["input_ids"].shape)

print("\n")

print("=" * 80)
print("ATTENTION MASK")
print("=" * 80)
print(encoded["attention_mask"])

print("\n")

print("=" * 80)
print("NUMBER OF TOKENS")
print("=" * 80)
print(encoded["input_ids"].shape[1])

print("\n")

print("=" * 80)
print("FIRST 30 TOKENS")
print("=" * 80)
print(encoded["input_ids"][0][:30])