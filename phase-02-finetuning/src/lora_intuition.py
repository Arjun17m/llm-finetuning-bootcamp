import torch

hidden_size = 2048

# Original q_proj weight
W = torch.empty(hidden_size, hidden_size)

print("=" * 60)
print("Original Weight Matrix")
print("=" * 60)

print("Shape :", W.shape)
print("Parameters :", W.numel())

print()

memory_mb = W.numel() * 2 / (1024**2)  # float16 = 2 bytes
print(f"Memory (float16): {memory_mb:.2f} MB")  