"""Utilities for wrapping."""
import numpy as np

def is_sequence(argument):
    """Check if argument is a sequence."""
    return (
        not hasattr(argument, "strip") and
        not hasattr(argument, "shape") and
        (hasattr(argument, "__getitem__") or hasattr(argument, "__iter__"))
    )


def force_to_column_major_and_double_precision(matrix):
    """
    Force given matrix to column major and double precision.

    Performance-wise it can help to avoid copies
    when wrapping C++ in python.
    """
    return np.array(matrix, order='F', dtype=np.float64)
