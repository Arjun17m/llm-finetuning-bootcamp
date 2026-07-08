"""
===============================================================================
Project : LLM Fine-Tuning Bootcamp

File:
    train.py

Purpose:
    Loads the processed dataset and tokenizer.

    This is the first step of the training pipeline.
===============================================================================
"""

# =============================================================================
# Hugging Face Dataset Library
# =============================================================================
from datasets import load_dataset

# =============================================================================
# Hugging Face Tokenizer
# =============================================================================
from transformers import AutoTokenizer

# =============================================================================
# Project Configuration
# =============================================================================
from config import Config


def load_processed_dataset():
    """
    Load the processed dataset created by prepare_dataset.py.
    """

    print("=" * 80)
    print("Loading Processed Dataset")
    print("=" * 80)

    dataset = load_dataset(
        "json",
        data_files=str(Config.PROCESSED_DATASET),
    )

    return dataset


def load_tokenizer():
    """
    Load the tokenizer that belongs to the model.
    """

    print("\nLoading Tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME
    )

    return tokenizer

def inspect_chat_template(dataset, tokenizer):
    """
    Inspect how the tokenizer converts a conversation
    into the text that the model was originally trained on.
    """

    print("\n" + "=" * 80)
    print("Inspecting Chat Template")
    print("=" * 80)

    # ------------------------------------------------------------
    # Get the first conversation from the dataset.
    # ------------------------------------------------------------
    messages = dataset["train"][0]["messages"]

    # ------------------------------------------------------------
    # Convert the conversation into the prompt format expected
    # by Llama.
    #
    # tokenize=False means:
    # Return plain text instead of token IDs.
    # ------------------------------------------------------------
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    print(prompt)


def inspect_tokenization(dataset, tokenizer):
    """
    Inspect how the tokenizer converts a conversation into tensors.

    Input:
        messages

    Output:
        input_ids
        attention_mask
    """

    print("\n" + "=" * 80)
    print("Inspecting Tokenization")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Get the first conversation from the processed dataset.
    # -----------------------------------------------------------------
    messages = dataset["train"][0]["messages"]

    # -----------------------------------------------------------------
    # Convert the conversation into the exact prompt format used by
    # the model during instruction tuning.
    # -----------------------------------------------------------------
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    print("\nFormatted Prompt\n")
    print(prompt)

    # -----------------------------------------------------------------
    # Convert the prompt into PyTorch tensors.
    #
    # return_tensors="pt"
    # tells the tokenizer to return torch.Tensor instead of Python lists.
    # -----------------------------------------------------------------
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )

    print("\n" + "=" * 80)
    print("INPUT IDS")
    print("=" * 80)

    print(encoded["input_ids"])

    print("\nShape:", encoded["input_ids"].shape)

    print("\n" + "=" * 80)
    print("ATTENTION MASK")
    print("=" * 80)

    print(encoded["attention_mask"])

    print("\nShape:", encoded["attention_mask"].shape)

    print("\n" + "=" * 80)
    print("NUMBER OF TOKENS")
    print("=" * 80)

    print(encoded["input_ids"].shape[1])

    # -----------------------------------------------------------------
    # Decode back to verify nothing changed during tokenization.
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DECODED BACK")
    print("=" * 80)

    decoded = tokenizer.decode(
        encoded["input_ids"][0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    print(decoded)

def main():

    dataset = load_processed_dataset()

    tokenizer = load_tokenizer()

    print("\nDataset Summary")
    print(dataset)

    print("\nColumns")
    print(dataset["train"].column_names)

    print("\nFirst Example")
    print(dataset["train"][0])

    print("\nTokenizer")

    print(type(tokenizer))

    print(tokenizer.__class__.__name__)

    print("\nEOS Token")

    print(tokenizer.eos_token)

    print("\nPAD Token")

    print(tokenizer.pad_token)
    inspect_chat_template(
                dataset,
                tokenizer,
        )

    inspect_tokenization(
    dataset,
    tokenizer,
        )
if __name__ == "__main__":
    main()