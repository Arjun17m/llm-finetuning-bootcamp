import torch
from transformers import AutoModelForCausalLM

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="cpu",   # CPU is enough for inspection
)

print("\nFirst 30 parameter names:\n")

for i, (name, param) in enumerate(model.named_parameters()):
    print(
        f"{i+1:2}. {name:55} "
        f"Shape={tuple(param.shape)} "
        f"Dtype={param.dtype}"
    )

    if i == 29:
        break

print("\nTotal Parameters:")
print(sum(p.numel() for p in model.parameters()))