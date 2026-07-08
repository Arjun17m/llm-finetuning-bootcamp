"""
Production Fine-Tuning Pipeline

Author: Arjun Pawar
Project: LLM Fine-Tuning Bootcamp
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from config import Config
from trl import SFTTrainer
from transformers import TrainingArguments



def formatting_func(example):

    messages = [
        {
            "role": "user",
            "content": f"{example['instruction']}\n\n{example['input']}",
        },
        {
            "role": "assistant",
            "content": example["output"],
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

def main():

    print("=" * 80)
    print("LLM Fine-Tuning Pipeline")
    print("=" * 80)

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    print("\nLoading Dataset...")

    dataset = load_dataset(
        "json",
        data_files=Config.DATASET_PATH,
    )

    print(dataset)

    # -------------------------------------------------
    # Tokenizer
    # -------------------------------------------------

    print("\nLoading Tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    print("\nLoading Model...")

    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )

    # -------------------------------------------------
    # LoRA
    # -------------------------------------------------

    print("\nApplying LoRA...")

    lora_config = LoraConfig(

        r=Config.LORA_R,

        lora_alpha=Config.LORA_ALPHA,

        lora_dropout=Config.LORA_DROPOUT,

        target_modules=Config.TARGET_MODULES,

        bias="none",

        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    print("\nTrainable Parameters")

    model.print_trainable_parameters()

    print("\nPipeline Ready")
    
    training_args = TrainingArguments(
                output_dir=Config.OUTPUT_DIR,
                num_train_epochs=Config.NUM_EPOCHS,
                per_device_train_batch_size=Config.BATCH_SIZE,
                learning_rate=Config.LEARNING_RATE,
                logging_steps=1,
                save_strategy="epoch",
                report_to="none",
                fp16=True,
            )

    trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,)

    print("=" * 80)
    print("Trainer Created Successfully")
    print("=" * 80)

    print(type(trainer))

    print()

    print("Dataset Size:", len(trainer.train_dataset))

    print()

    print("Model Type:", type(trainer.model))
if __name__ == "__main__":

    main()