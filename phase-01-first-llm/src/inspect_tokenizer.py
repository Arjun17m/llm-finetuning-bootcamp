from transformers import AutoTokenizer

MODEL = "unsloth/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("=" * 80)

print("Type:")
print(type(tokenizer))

print("=" * 80)

print("Tokenizer class:")
print(tokenizer.__class__)

print("=" * 80)

print("Tokenizer class name:")
print(tokenizer.__class__.__name__)

print("=" * 80)

print("Backend tokenizer:")
print(tokenizer.backend_tokenizer)

print("=" * 80)