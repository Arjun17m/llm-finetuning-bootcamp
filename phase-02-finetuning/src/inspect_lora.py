import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="cpu",
)

print("\nTrainable parameters BEFORE LoRA")

model.print_trainable_parameters = lambda: print(
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

model.print_trainable_parameters()

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

print("\nTrainable parameters AFTER LoRA")

model.print_trainable_parameters()