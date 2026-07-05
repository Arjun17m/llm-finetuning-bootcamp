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
        "content": "Repeat the word 'Transformer' twenty times."
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

import time
start = time.time()

inputs = tokenizer(text, return_tensors="pt").to(model.device)
print("=" * 50)
print("EOS Token      :", tokenizer.eos_token)
print("EOS Token ID   :", tokenizer.eos_token_id)
print("PAD Token      :", tokenizer.pad_token)
print("PAD Token ID   :", tokenizer.pad_token_id)
print("=" * 50)


outputs = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.7,
    top_p=1.0,
    top_k=100,
    repetition_penalty=2.0,
    max_new_tokens=150,
)
generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

print("\nGenerated Token IDs:")
print(generated_tokens.tolist())

print("\nDecoded (with special tokens):")
print(tokenizer.decode(generated_tokens, skip_special_tokens=False))

print("\nDecoded (without special tokens):")
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))

end = time.time()

print(f"Generation Time: {end - start:.2f} seconds")
# print("\n==========================")
# # print(response)
# print("==========================")