import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

print(f"Loading tokenizer: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading model: {MODEL_NAME}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": "Write an original science fiction story about AI discovering an ancient civilization on Mars. Make it surprising."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("=" * 80)
print(text)
print("=" * 80)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

generation_config = model.generation_config.clone()

generation_config.do_sample = True
generation_config.temperature = 0.7
generation_config.top_p = 0.5

outputs = model.generate(
    **inputs,
    generation_config=generation_config,
    max_new_tokens=80,
)
print("\n==========================")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("==========================")