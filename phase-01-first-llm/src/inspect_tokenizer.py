from transformers import AutoTokenizer

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("=" * 60)
print("Tokenizer Class :", tokenizer.__class__.__name__)
print("Vocabulary Size :", tokenizer.vocab_size)
print("Model Max Length:", tokenizer.model_max_length)
print("Padding Side    :", tokenizer.padding_side)
print("Truncation Side :", tokenizer.truncation_side)
print("Special Tokens  :", tokenizer.special_tokens_map)
print("=" * 60)