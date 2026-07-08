"""
===============================================================================
Project : LLM Fine-Tuning Bootcamp
Sprint  : 2 - Production Dataset Pipeline

File:
    config.py

Purpose:
    Central configuration file for the complete fine-tuning pipeline.

    Every configurable value should live here instead of being
    hardcoded throughout the project.

Author:
    Arjun Pawar
===============================================================================
"""

from pathlib import Path


class Config:
    """
    Stores every configuration used across the project.

    Instead of writing paths and hyperparameters inside multiple files,
    every script imports this class.

    Example:
        from config import Config

        print(Config.MODEL_NAME)
    """

    # =====================================================================
    # Project Root
    # =====================================================================

    # Resolve the absolute path of the current file.
    # Example:
    # C:/.../phase-02-finetuning/src/config.py
    CURRENT_FILE = Path(__file__).resolve()

    # src/
    SRC_DIR = CURRENT_FILE.parent

    # phase-02-finetuning/
    PROJECT_DIR = SRC_DIR.parent

    # =====================================================================
    # Dataset
    # =====================================================================

    DATA_DIR = PROJECT_DIR / "data"

    RAW_DATA_DIR = DATA_DIR / "raw"

    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    RAW_DATASET = RAW_DATA_DIR / "instruction_dataset.json"

    PROCESSED_DATASET = PROCESSED_DATA_DIR / "train.json"

    # =====================================================================
    # Model
    # =====================================================================

    MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

    MAX_SEQ_LENGTH = 512

    # =====================================================================
    # LoRA
    # =====================================================================

    LORA_R = 8

    LORA_ALPHA = 16

    LORA_DROPOUT = 0.05

    TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]

    # =====================================================================
    # Training
    # =====================================================================

    OUTPUT_DIR = PROJECT_DIR / "outputs"

    CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

    BATCH_SIZE = 1

    LEARNING_RATE = 2e-4

    NUM_EPOCHS = 3

    RANDOM_SEED = 42

    # =====================================================================
    # Logging
    # =====================================================================

    LOG_LEVEL = "INFO"