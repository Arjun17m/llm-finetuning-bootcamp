"""
===============================================================================
Project : LLM Fine-Tuning Bootcamp

File:
    inference.py

Purpose:
    Load a trained LoRA adapter and generate responses.

===============================================================================
"""


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel

import torch

from config import Config



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

def load_adapter(model):
    """
    Load the trained LoRA adapter.
    """

    model = PeftModel.from_pretrained(

        model,

        Config.OUTPUT_DIR / "checkpoint-3",

    )

    return model

def generate_response(
    model,
    tokenizer,
    prompt,
):
    """
    Generate a response from the fine-tuned model.
    """

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

    prompt_length = inputs["input_ids"].shape[-1]

    response = tokenizer.decode(
        outputs[0][prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return response

def main():
    """
    Main function to load the model, tokenizer, and adapter.
    """

    print("\nLoading Tokenizer...\n")
    tokenizer = load_tokenizer()

    print("\nLoading Base Model...\n")
    model = load_model()

    print("\nLoading LoRA Adapter...\n")
    model = load_adapter(model)

    print("\nModel and Tokenizer Loaded Successfully!\n")
    print()

    print()

    response = generate_response(
        model,
        tokenizer,
        "What is Artificial Intelligence?"
    )

    print(response)

if __name__ == "__main__":
    main()