"""Python wrapper for UMKLKNN."""
import numpy as np
from ._pymimkl import UMKLKNN_
from .utils import (
    is_sequence, force_to_column_major_and_double_precision
)


class UMKLKNN(UMKLKNN_):
    """Wrapping UMKLKNN_."""

    def fit(self, X, y=None):
        """
        Fit model.

        Fit given a data matrix (numpy ndarray format) or
        a list of kernel matrices (numpy ndarray format).
        """
        if issubclass(type(X), np.ndarray):
            X = force_to_column_major_and_double_precision(X)
        elif is_sequence(X):
            X = [
                force_to_column_major_and_double_precision(a_kernel)
                for a_kernel in X
            ]
        else:
            RuntimeError(
                'X must be a numpy ndarray or a list of numpy ndarray'
            )
        UMKLKNN_.fit(self, X)
        self.kernels_weights = self.beta
        return self

    def predict(self, X):
        """
        Predict labels.

        Predict given a data matrix (numpy ndarray format) or
        a list of kernel matrices (numpy ndarray format).
        """
        if X is not None:
            if issubclass(type(X), np.ndarray):
                X = force_to_column_major_and_double_precision(X)
            elif is_sequence(X):
                X = [
                    force_to_column_major_and_double_precision(a_kernel)
                    for a_kernel in X
                ]
        # predict
        return UMKLKNN_.predict(self, X)
