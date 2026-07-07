from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = load_dataset(
    "json",
    data_files="../data/instruction_dataset.json"
)

example = dataset["train"][0]

print("\nOriginal Dataset Example\n")
print(example)

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

print("\n")
print("=" * 80)
print("FORMATTED PROMPT")
print("=" * 80)
print(formatted_prompt)
print("=" * 80)