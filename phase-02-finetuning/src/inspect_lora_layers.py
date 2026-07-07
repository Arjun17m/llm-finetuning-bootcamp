import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="cpu",
)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

print("=" * 80)
print("LoRA Modules")
print("=" * 80)

count = 0

for name, module in model.named_modules():

    if "lora" in name.lower():
        print(name)
        count += 1

print()
print("=" * 80)
print("Total LoRA modules:", count)
print("=" * 80)