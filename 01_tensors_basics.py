import torch

# ── 1. System Info ──────────────────────────────────────────────
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print()

# ── 2. Tensors from Python lists ────────────────────────────────
# 1D tensor (vector)
data_1d = [1, 2, 3, 4, 5]
tensor_1d = torch.tensor(data_1d)
print("1D Tensor:", tensor_1d)
print("   shape:", tensor_1d.shape, " dtype:", tensor_1d.dtype)
print()

# 2D tensor (matrix)
data_2d = [[1, 2, 3],
           [4, 5, 6]]
tensor_2d = torch.tensor(data_2d)
print("2D Tensor:\n", tensor_2d)
print("   shape:", tensor_2d.shape)
print()

# 3D tensor
data_3d = [[[1, 2], [3, 4]],
           [[5, 6], [7, 8]]]
tensor_3d = torch.tensor(data_3d)
print("3D Tensor:\n", tensor_3d)
print("   shape:", tensor_3d.shape)
print()

# ── 3. Tensors with specific dtypes ────────────────────────────
float_tensor = torch.tensor([1.0, 2.5, 3.7])
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
bool_tensor = torch.tensor([True, False, True])

print("Float tensor:", float_tensor, "  dtype:", float_tensor.dtype)
print("Int tensor:  ", int_tensor, "  dtype:", int_tensor.dtype)
print("Bool tensor: ", bool_tensor, "  dtype:", bool_tensor.dtype)
print()

# ── 4. Common tensor creation functions ─────────────────────────
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
rand = torch.rand(2, 3)       # uniform [0, 1)
randn = torch.randn(2, 3)     # normal distribution

print("Zeros:\n", zeros)
print("Ones:\n", ones)
print("Random [0,1):\n", rand)
print("Random normal:\n", randn)
print()

# ── 5. Tensor from NumPy array ──────────────────────────────────
import numpy as np

np_array = np.array([10, 20, 30, 40, 50])
tensor_from_np = torch.from_numpy(np_array)
print("NumPy array: ", np_array)
print("Tensor:      ", tensor_from_np)
print()

# ── 6. Move tensor to GPU (if available) ────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_tensor = tensor_2d.to(device)
print(f"Tensor on {device}:\n", gpu_tensor)
print()

# ── 7. Basic operations ────────────────────────────────────────
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("a + b =", a + b)
print("a * b =", a * b)           # element-wise
print("dot product =", torch.dot(a, b))
print("mean =", a.mean())
print("sum  =", a.sum())
