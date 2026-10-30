"""Unit tests for Pure Python List linear algebra operations."""

import unittest

from listai.engines.list_engine.operations import (
    dot_product,
    matrix_add,
    matrix_add_bias,
    matrix_elementwise_mul,
    matrix_mean,
    matrix_mul,
    matrix_scalar_mul,
    matrix_shape,
    matrix_sub,
    matrix_sum_columns,
    matrix_vector_mul,
    matrix_zeros,
    transpose,
    vector_add,
    vector_mean,
    vector_scalar_mul,
    vector_sub,
    vector_zeros,
)


class TestListOperations(unittest.TestCase):
    """Test manual vector and matrix operations implemented with Python Lists."""

    def test_vector_and_matrix_creation(self) -> None:
        """Verify vector and matrix allocation with zeros."""
        v = vector_zeros(5)
        self.assertEqual(v, [0.0, 0.0, 0.0, 0.0, 0.0])

        M = matrix_zeros(2, 3)
        self.assertEqual(M, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertEqual(matrix_shape(M), (2, 3))

        with self.assertRaises(ValueError):
            matrix_shape([[1.0, 2.0], [3.0]])  # Ragged matrix

    def test_vector_arithmetic(self) -> None:
        """Verify vector addition, subtraction, and scalar multiplication."""
        v1 = [1.0, -2.0, 3.5]
        v2 = [4.0, 2.0, -1.5]

        self.assertEqual(vector_add(v1, v2), [5.0, 0.0, 2.0])
        self.assertEqual(vector_sub(v1, v2), [-3.0, -4.0, 5.0])
        self.assertEqual(vector_scalar_mul(v1, 2.0), [2.0, -4.0, 7.0])

        with self.assertRaises(ValueError):
            vector_add([1.0], [1.0, 2.0])

    def test_dot_product_and_matrix_vector(self) -> None:
        """Verify dot product and matrix-vector multiplication."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, -5.0, 6.0]
        # 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12
        self.assertEqual(dot_product(v1, v2), 12.0)

        M = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
        # M (2x3) @ v (3,) = [1*1 + 2*0 + 3*(-1), 4*1 + 5*0 + 6*(-1)] = [-2.0, -2.0]
        v = [1.0, 0.0, -1.0]
        self.assertEqual(matrix_vector_mul(M, v), [-2.0, -2.0])

    def test_matrix_multiplication(self) -> None:
        """Verify matrix multiplication A @ B and identity property."""
        A = [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
        I = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        self.assertEqual(matrix_mul(A, I), A)

        # (2x3) x (3x2) -> (2x2)
        A2 = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
        B2 = [
            [7.0, 8.0],
            [9.0, 1.0],
            [2.0, 3.0],
        ]
        # row 0: [1*7+2*9+3*2, 1*8+2*1+3*3] = [7+18+6, 8+2+9] = [31, 19]
        # row 1: [4*7+5*9+6*2, 4*8+5*1+6*3] = [28+45+12, 32+5+18] = [85, 55]
        expected = [
            [31.0, 19.0],
            [85.0, 55.0],
        ]
        self.assertEqual(matrix_mul(A2, B2), expected)

        with self.assertRaises(ValueError):
            matrix_mul([[1.0, 2.0]], [[1.0, 2.0]])  # (1, 2) x (1, 2) incompatible

    def test_transpose(self) -> None:
        """Verify matrix transposition and involution (A^T)^T == A."""
        M = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
        M_T = transpose(M)
        self.assertEqual(
            M_T,
            [
                [1.0, 4.0],
                [2.0, 5.0],
                [3.0, 6.0],
            ],
        )
        self.assertEqual(transpose(M_T), M)

    def test_matrix_arithmetic_and_bias(self) -> None:
        """Verify matrix add, sub, scalar mul, Hadamard product, and bias broadcast."""
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]

        self.assertEqual(matrix_add(A, B), [[6.0, 8.0], [10.0, 12.0]])
        self.assertEqual(matrix_sub(A, B), [[-4.0, -4.0], [-4.0, -4.0]])
        self.assertEqual(matrix_scalar_mul(A, 0.5), [[0.5, 1.0], [1.5, 2.0]])
        self.assertEqual(matrix_elementwise_mul(A, B), [[5.0, 12.0], [21.0, 32.0]])

        # Bias addition: M (2x2) + b (2,)
        bias = [10.0, 20.0]
        self.assertEqual(matrix_add_bias(A, bias), [[11.0, 22.0], [13.0, 24.0]])

    def test_reductions(self) -> None:
        """Verify column sums and mean calculations."""
        M = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
        self.assertEqual(matrix_sum_columns(M), [5.0, 7.0, 9.0])
        self.assertEqual(matrix_mean(M), 3.5)
        self.assertEqual(vector_mean([1.0, 2.0, 3.0, 4.0]), 2.5)


if __name__ == "__main__":
    unittest.main()
