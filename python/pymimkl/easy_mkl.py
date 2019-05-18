"""Python wrapper for EasyMKL."""
import numpy as np
from ._pymimkl import EasyMKL_
from .utils import is_sequence, force_to_column_major_and_double_precision


def translate_y_type(self, labels):
    """
    Translating the class annotation from type str to the original type.
    On the c++ side, the labels needed to be of type str.
    """
    if self.y_type is bool or self.y_type is np.bool_:
        return np.array([
            label == 'True' for label in labels
        ])
    else:
        return np.array([
            self.y_type(label) for label in labels
        ])


class EasyMKL(EasyMKL_):
    """Wrapping EasyMKL_."""
    y_type = None

    def fit(self, X, y):
        """
        Fit model.

        Fit given a data matrix (numpy ndarray format) or
        a list of kernel matrices (numpy ndarray format).
        """
        if issubclass(type(y), np.ndarray) or is_sequence(y):
            self.y_type = type(y[0])
            y = [str(label) for label in y]
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
            EasyMKL_.fit(self, X, y)
            self.kernels_weights = self.etas
            return self
        else:
            raise RuntimeError('y must be a a list or a numpy ndarray')

    def predict(self, X):
        """
        Predict labels.

        Predict given a data matrix (numpy ndarray format) or
        a list of kernel matrices (numpy ndarray format).
        """
        if issubclass(type(X), np.ndarray):
            X = force_to_column_major_and_double_precision(X)
        elif is_sequence(X):
            X = [
                force_to_column_major_and_double_precision(a_kernel)
                for a_kernel in X
            ]
        # predict
        return translate_y_type(self, EasyMKL_.predict(self, X))

    def get_classes_order(self):
        """
        return correct type of class labels in order of
        the binary 1 vs Rest problems in kernel_weights
        """
        return translate_y_type(self, self.get_one_versus_rest_order())
