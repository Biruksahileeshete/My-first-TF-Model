# ========================================
# BUILD My FIRST TENSORFLOW MODEL
# Complete Beginner Guide
# ========================================

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")

# ========================================
# PART 1: TENSOR BASICS
# ========================================

print("\n" + "="*50)
print("PART 1: TENSOR BASICS")
print("="*50)

# 1. Create tensors
print("\n📊 Creating tensors:")

# Scalar (0D tensor)
scalar = tf.constant(5)
print(f"Scalar: {scalar}")

# Vector (1D tensor)
vector = tf.constant([1, 2, 3, 4, 5])
print(f"Vector: {vector}")

# Matrix (2D tensor)
matrix = tf.constant([[1, 2], [3, 4]])
print(f"Matrix:\n{matrix}")

# 3D tensor
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D Tensor shape: {tensor_3d.shape}")

# 2. Tensor properties
print("\n📐 Tensor Properties:")
print(f"Shape: {matrix.shape}")
print(f"Data type: {matrix.dtype}")
print(f"Size: {tf.size(matrix)}")

# 3. Tensor operations
print("\n🔢 Tensor Operations:")
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

print(f"Addition: {a + b}")
print(f"Subtraction: {a - b}")
print(f"Multiplication: {a * b}")
print(f"Division: {a / b}")
print(f"Dot product: {tf.tensordot(a, b, axes=1)}")

# 4. Creating special tensors
print("\n🎯 Special Tensors:")
print(f"Zeros: {tf.zeros([2, 3])}")
print(f"Ones: {tf.ones([2, 3])}")
print(f"Identity: {tf.eye(3)}")
print(f"Random: {tf.random.normal([2, 3])}")
