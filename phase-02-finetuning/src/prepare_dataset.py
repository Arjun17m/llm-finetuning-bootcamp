"""
===============================================================================
Project : LLM Fine-Tuning Bootcamp

File:
    prepare_dataset.py

Purpose:
    Loads the raw instruction dataset and validates that it has
    the required columns before any preprocessing.

Author:
    Arjun Pawar
===============================================================================
"""

# =============================================================================
# Import the Hugging Face dataset library.
# It allows us to load datasets from JSON, CSV, Parquet and the HF Hub.
# =============================================================================
from datasets import load_dataset

# =============================================================================
# Import the central configuration for the project.
# This avoids hardcoding file paths throughout the codebase.
# =============================================================================
from config import Config


def load_raw_dataset():
    """
    Load the raw dataset from disk.

    Returns:
        DatasetDict
    """

    print("=" * 80)
    print("Loading Raw Dataset")
    print("=" * 80)

    dataset = load_dataset(
        "json",
        data_files=str(Config.RAW_DATASET),
    )

    return dataset


def validate_dataset(dataset):
    """
    Validate that the dataset contains the required columns.
    """

    print("\nValidating Dataset...")

    required_columns = {
        "instruction",
        "input",
        "output",
    }

    actual_columns = set(dataset["train"].column_names)

    missing = required_columns - actual_columns

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("✓ Dataset validation successful")

def convert_to_messages(example):
    """
    Convert one raw dataset example into a conversation.

    Input:
        {
            "instruction": "...",
            "input": "...",
            "output": "..."
        }

    Output:
        {
            "messages": [...]
        }
    """

    # -------------------------------------------------------------
    # Start with the instruction.
    # -------------------------------------------------------------
    user_prompt = example["instruction"]

    # -------------------------------------------------------------
    # Some datasets have additional input/context.
    # If it exists, append it to the instruction.
    # -------------------------------------------------------------
    if example["input"].strip():

        user_prompt += "\n\n" + example["input"]

    # -------------------------------------------------------------
    # Build the conversation.
    # This is the format expected by modern chat models.
    # -------------------------------------------------------------
    messages = [

        {
            "role": "user",
            "content": user_prompt,
        },

        {
            "role": "assistant",
            "content": example["output"],
        },
    ]

    return {
        "messages": messages
    }

def main():

    dataset = load_raw_dataset()

    validate_dataset(dataset)

    print("\n" + "=" * 80)
    print("Converting Complete Dataset")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Apply our conversion function to every row in the training dataset.
    #
    # The original dataset is NOT modified.
    # map() returns a NEW dataset containing the transformed rows.
    # -------------------------------------------------------------------------
    formatted_dataset = dataset["train"].map(
        convert_to_messages
    )

    formatted_dataset = dataset["train"].map(
    convert_to_messages,

    # ------------------------------------------------------------
    # After creating the "messages" column,
    # remove the original columns because they are no longer needed
    # for training.
    # ------------------------------------------------------------
    remove_columns=dataset["train"].column_names,)
    print("\nDataset After Conversion")
    print(formatted_dataset)

    print("formatted dataset columns", formatted_dataset.column_names)

    print("\nFirst Converted Example")
    print(formatted_dataset[0])
    # --------------------------------------------------------------------
    # Save the processed dataset to disk.
    #
    # Instead of repeating preprocessing every time we train,
    # we save the processed dataset once and reuse it.
    # --------------------------------------------------------------------

    print("\nSaving Processed Dataset...")

    formatted_dataset.to_json(
        str(Config.PROCESSED_DATASET)
    )

    print("✓ Dataset saved successfully")

    print(Config.PROCESSED_DATASET)

if __name__ == "__main__":
    main()