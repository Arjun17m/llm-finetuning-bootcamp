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
from xml.parsers.expat import model

from datasets import load_dataset

# =============================================================================
# Hugging Face Tokenizer
# =============================================================================
from transformers import AutoTokenizer

# =============================================================================
# Project Configuration
# =============================================================================
from config import Config
# =============================================================================
# PyTorch is used because the model returns torch.Tensor objects.
# =============================================================================
import torch

# =============================================================================
# AutoModelForCausalLM loads decoder-only language models such as Llama.
# =============================================================================
from transformers import AutoModelForCausalLM

# =============================================================================
# Functional contains activation functions like softmax.
# =============================================================================
import torch.nn.functional as F
import math


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

def load_model():
    """
    Load the pretrained Llama model.

    At this stage we are NOT applying LoRA.
    We simply load the original pretrained model.
    """

    print("\nLoading Model...")

    model = AutoModelForCausalLM.from_pretrained(

        Config.MODEL_NAME,

        # Use float16 to reduce GPU memory usage.
        torch_dtype=torch.float16,

        # Automatically place the model on the GPU.
        device_map="auto",
    )

    return model

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


def inspect_forward_pass(dataset, tokenizer, model):
    """
    Run one training example through the model.

    We are NOT training.
    We are simply inspecting the model output.
    """

    print("\n" + "=" * 80)
    print("FORWARD PASS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Get the first conversation.
    # ------------------------------------------------------------------
    messages = dataset["train"][0]["messages"]

    # ------------------------------------------------------------------
    # Convert conversation into Llama prompt.
    # ------------------------------------------------------------------
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # ------------------------------------------------------------------
    # Convert text into tensors.
    # ------------------------------------------------------------------
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )

    # ------------------------------------------------------------------
    # Move tensors to the same device as the model.
    # ------------------------------------------------------------------
    encoded = {
        key: value.to(model.device)
        for key, value in encoded.items()
    }

    # ------------------------------------------------------------------
    # Disable gradient computation.
    # We are only inspecting the output.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # During training we provide labels.
    #
    # AutoModelForCausalLM automatically:
    #
    # 1. Shifts labels
    # 2. Computes Cross Entropy Loss
    # 3. Returns the scalar loss
    # ------------------------------------------------------------------

    labels = encoded["input_ids"].clone()

    outputs = model(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"],
    labels=labels,
    )

    print("\nOutput Type")
    print(type(outputs))

    print("\nAvailable Keys")
    print(outputs.keys())

    print("\nLogits Shape")
    print(outputs.logits.shape)

    print("\nLoss")

    print(outputs.loss)
    print("\nLoss Type")

    print(type(outputs.loss))
    print("\nRunning Backward Pass...")

    outputs.loss.backward()

    print("Done")
    print("\nGradient Inspection")

    print("="*80)

    for name, parameter in model.named_parameters():

        if parameter.grad is not None:

            print(name)

            print(parameter.grad.shape)

            break
    # print("\nFirst Token Logits Shape")
    # print(outputs.logits[0, 0].shape)

    # print("\nFirst Five Logits")
    # print(outputs.logits[0, 0, :5])

    # # ------------------------------------------------------------------
    # # Convert logits into probabilities.
    # #
    # # Softmax transforms arbitrary scores into probabilities whose sum
    # # equals 1.
    # # ------------------------------------------------------------------

    # probabilities = F.softmax(
    #     outputs.logits[0, 0],
    #     dim=-1,
    # )

    # print("\n" + "=" * 80)
    # print("MANUAL LOSS")
    # print("=" * 80)

    # # ---------------------------------------------------------
    # # The correct next token for the FIRST position.
    # #
    # # Remember:
    # #
    # # Input:
    # # BOS  What  is ...
    # #
    # # Label:
    # # What is Artificial ... 
    # #
    # # For today we simply inspect the token at position 1.
    # # ---------------------------------------------------------

    # correct_token = encoded["input_ids"][0, 1]

    # print("Correct Token ID")

    # print(correct_token)

    # correct_probability = probabilities[correct_token]

    # print("\nProbability Assigned To Correct Token")

    # print(correct_probability)

    # manual_loss = -math.log(
    #     float(correct_probability)
    # )

    # print("\nManual Loss")

    # print(manual_loss)
def main():

    dataset = load_processed_dataset()

    tokenizer = load_tokenizer()

    model = load_model()

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
    
    print("\nModel")

    print(type(model))

    print()

    print(model.__class__.__name__)

    print()

    print("Device")

    print(model.device)

    print()

    print("Training Mode")

    print(model.training)

    inspect_forward_pass(
        dataset,
        tokenizer,
        model,
    )




if __name__ == "__main__":
    main()