"""
===============================================================================
Project : LLM Fine-Tuning Bootcamp

File:
    train.py

Purpose:
    Fine-tune a Llama model using LoRA.

This file contains only the production training pipeline.

Debugging and learning code lives in model_inspector.py
===============================================================================
"""

# =============================================================================
# Hugging Face Dataset Library
# =============================================================================
from datasets import load_dataset

# =============================================================================
# Hugging Face Transformers
# =============================================================================
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
# =============================================================================
# PEFT
# =============================================================================
from peft import (
    LoraConfig,
    get_peft_model,
)
# =============================================================================
# PyTorch
# =============================================================================
import torch

# =============================================================================
# Project Configuration
# =============================================================================
from config import Config

from trl import SFTConfig , SFTTrainer


def load_processed_dataset():
    """
    Load the processed training dataset.
    """

    dataset = load_dataset(
        "json",
        data_files=str(Config.PROCESSED_DATASET),
    )

    return dataset

def load_tokenizer():
    """
    Load the tokenizer associated with the base model.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model():
    """
    Load the pretrained Llama model.
    """

    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )

    return model

def apply_lora(model):
    """
    Apply LoRA adapters to the pretrained model.
    """

    peft_config = LoraConfig(

        # Rank of LoRA matrices
        r=Config.LORA_R,

        # Scaling factor
        lora_alpha=Config.LORA_ALPHA,

        # Dropout applied during training
        lora_dropout=Config.LORA_DROPOUT,

        # Modules where LoRA will be injected
        target_modules=Config.TARGET_MODULES,

        # Don't train bias parameters
        bias="none",

        # Decoder-only language model
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    return model

def create_training_config():
    """
    Create the training configuration for supervised fine-tuning.
    """

    training_config = SFTConfig(

        output_dir=Config.OUTPUT_DIR,

        num_train_epochs=Config.NUM_EPOCHS,

        per_device_train_batch_size=Config.BATCH_SIZE,

        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,

        learning_rate=Config.LEARNING_RATE,

        logging_steps=Config.LOGGING_STEPS,

        save_strategy="epoch",

        report_to="none",
    )

    return training_config

def create_trainer(
    model,
    tokenizer,
    dataset,
    training_config,
):
    """
    Create the Supervised Fine-Tuning trainer.
    """

    trainer = SFTTrainer(

        model=model,

        processing_class=tokenizer,

        train_dataset=dataset["train"],

        args=training_config,
    )

    return trainer

def main():

    dataset = load_processed_dataset()

    tokenizer = load_tokenizer()

    model = load_model()

    model = apply_lora(model)

    model.print_trainable_parameters()

    training_config = create_training_config()

    trainer = create_trainer(
    model,
    tokenizer,
    dataset,
    training_config,    )

    print("\nStarting Training...\n")

    trainer.train()

    print("\nTraining Completed Successfully!")
    print("\nSaving Final Adapter...\n")

    trainer.save_model(
        Config.OUTPUT_DIR / "final_adapter"
    )

    print("Final Adapter Saved Successfully!")

if __name__ == "__main__":
    main()