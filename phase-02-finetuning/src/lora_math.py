import torch

hidden_size = 2048
rank = 8

# Original weight
W = torch.empty(hidden_size, hidden_size)

# LoRA matrices
A = torch.empty(rank, hidden_size)
B = torch.empty(hidden_size, rank)

print("=" * 60)
print("Original Weight")
print("=" * 60)
print("Shape:", W.shape)
print("Parameters:", W.numel())

print()

print("=" * 60)
print("LoRA Matrix A")
print("=" * 60)
print("Shape:", A.shape)
print("Parameters:", A.numel())

print()

print("=" * 60)
print("LoRA Matrix B")
print("=" * 60)
print("Shape:", B.shape)
print("Parameters:", B.numel())

print()

print("=" * 60)

lora_params = A.numel() + B.numel()

print("Total LoRA Parameters:", lora_params)

reduction = W.numel() / lora_params

print(f"Reduction Factor: {reduction:.1f}x")