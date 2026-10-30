"""Pure Python List linear algebra and matrix operations.

STRICT CONSTRAINT:
Only Python standard library is permitted. Zero imports of external numerical
libraries (NumPy, SciPy, Pandas, etc.).
"""

from typing import List, Sequence, Tuple


def vector_zeros(dim: int) -> List[float]:
    """Create a 1D vector of zeros with length `dim`."""
    if dim <= 0:
        raise ValueError(f"Vector dimension must be positive, got {dim}")
    return [0.0] * dim


def matrix_zeros(rows: int, cols: int) -> List[List[float]]:
    """Create a 2D matrix of zeros with shape (rows, cols)."""
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Matrix dimensions must be positive, got rows={rows}, cols={cols}")
    return [[0.0] * cols for _ in range(rows)]


def matrix_shape(M: Sequence[Sequence[float]]) -> Tuple[int, int]:
    """Return (rows, cols) shape of a 2D matrix."""
    if not M:
        return (0, 0)
    cols = len(M[0])
    for r_idx, row in enumerate(M):
        if len(row) != cols:
            raise ValueError(f"Ragged matrix: row 0 has {cols} cols, row {r_idx} has {len(row)}")
    return (len(M), cols)


def vector_add(v1: Sequence[float], v2: Sequence[float]) -> List[float]:
    """Element-wise addition of two 1D vectors."""
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimension mismatch: {len(v1)} vs {len(v2)}")
    return [a + b for a, b in zip(v1, v2)]


def vector_sub(v1: Sequence[float], v2: Sequence[float]) -> List[float]:
    """Element-wise subtraction of two 1D vectors: v1 - v2."""
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimension mismatch: {len(v1)} vs {len(v2)}")
    return [a - b for a, b in zip(v1, v2)]


def vector_scalar_mul(v: Sequence[float], scalar: float) -> List[float]:
    """Multiply every element in vector `v` by a scalar."""
    return [x * scalar for x in v]


def matrix_scalar_mul(M: Sequence[Sequence[float]], scalar: float) -> List[List[float]]:
    """Multiply every element in matrix `M` by a scalar."""
    return [[val * scalar for val in row] for row in M]


def dot_product(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Compute dot product of two 1D vectors: sum(v1_i * v2_i)."""
    if len(v1) != len(v2):
        raise ValueError(f"Vector length mismatch: {len(v1)} vs {len(v2)}")
    return sum(a * b for a, b in zip(v1, v2))


def matrix_vector_mul(M: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    """Compute matrix-vector multiplication M @ v, where M is (R, C) and v is (C,)."""
    rows, cols = matrix_shape(M)
    if cols != len(v):
        raise ValueError(f"Cannot multiply matrix ({rows}, {cols}) with vector of length {len(v)}")
    return [dot_product(row, v) for row in M]


def matrix_mul(
    A: Sequence[Sequence[float]],
    B: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Compute matrix multiplication A @ B, where A is (M, K) and B is (K, N).

    Returns a matrix of shape (M, N).
    """
    rows_A, cols_A = matrix_shape(A)
    rows_B, cols_B = matrix_shape(B)

    if cols_A != rows_B:
        raise ValueError(
            f"Shape mismatch for matrix multiplication: ({rows_A}, {cols_A}) x ({rows_B}, {cols_B})"
        )

    # Transpose B for row-wise dot products (significantly faster cache/pointer access in Python)
    B_T = transpose(B)
    result = [[dot_product(row_A, col_B) for col_B in B_T] for row_A in A]
    return result


def transpose(M: Sequence[Sequence[float]]) -> List[List[float]]:
    """Transpose a 2D matrix of shape (R, C) to (C, R)."""
    rows, cols = matrix_shape(M)
    if rows == 0 or cols == 0:
        return []
    return [[M[r][c] for r in range(rows)] for c in range(cols)]


def matrix_add(
    A: Sequence[Sequence[float]],
    B: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Element-wise addition of two matrices of identical shape."""
    shape_A = matrix_shape(A)
    shape_B = matrix_shape(B)
    if shape_A != shape_B:
        raise ValueError(f"Matrix shape mismatch: {shape_A} vs {shape_B}")
    return [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]


def matrix_sub(
    A: Sequence[Sequence[float]],
    B: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Element-wise subtraction of two matrices of identical shape: A - B."""
    shape_A = matrix_shape(A)
    shape_B = matrix_shape(B)
    if shape_A != shape_B:
        raise ValueError(f"Matrix shape mismatch: {shape_A} vs {shape_B}")
    return [[a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]


def matrix_elementwise_mul(
    A: Sequence[Sequence[float]],
    B: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Element-wise Hadamard product of two matrices: A * B."""
    shape_A = matrix_shape(A)
    shape_B = matrix_shape(B)
    if shape_A != shape_B:
        raise ValueError(f"Matrix shape mismatch for Hadamard product: {shape_A} vs {shape_B}")
    return [[a * b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]


def matrix_add_bias(
    M: Sequence[Sequence[float]],
    b: Sequence[float],
) -> List[List[float]]:
    """Add 1D bias vector across each row of matrix M (batch broadcasting)."""
    rows, cols = matrix_shape(M)
    if cols != len(b):
        raise ValueError(f"Bias dimension {len(b)} does not match matrix columns {cols}")
    return [[val + bias_val for val, bias_val in zip(row, b)] for row in M]


def matrix_sum_columns(M: Sequence[Sequence[float]]) -> List[float]:
    """Sum across rows for each column.

    Used to compute bias gradients: d_b = sum_i(dZ_i)
    """
    rows, cols = matrix_shape(M)
    if rows == 0:
        return []
    col_sums = [0.0] * cols
    for row in M:
        for c in range(cols):
            col_sums[c] += row[c]
    return col_sums


def vector_mean(v: Sequence[float]) -> float:
    """Compute arithmetic mean of a 1D vector."""
    if not v:
        raise ValueError("Cannot compute mean of empty vector")
    return sum(v) / len(v)


def matrix_mean(M: Sequence[Sequence[float]]) -> float:
    """Compute arithmetic mean of all elements in a 2D matrix."""
    rows, cols = matrix_shape(M)
    if rows == 0 or cols == 0:
        raise ValueError("Cannot compute mean of empty matrix")
    total = sum(sum(row) for row in M)
    return total / (rows * cols)
