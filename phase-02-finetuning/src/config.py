from dataclasses import dataclass

@dataclass
class Config:

    # ------------------------------
    # Model
    # ------------------------------

    MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

    # ------------------------------
    # Dataset
    # ------------------------------

    DATASET_PATH = "phase-02-finetuning/data/instruction_dataset.json"

    # ------------------------------
    # LoRA
    # ------------------------------

    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05

    TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]

    # ------------------------------
    # Training
    # ------------------------------

    OUTPUT_DIR = "phase-02-finetuning/checkpoints"

    BATCH_SIZE = 1

    LEARNING_RATE = 2e-4

    NUM_EPOCHS = 3

    MAX_SEQ_LENGTH = 512