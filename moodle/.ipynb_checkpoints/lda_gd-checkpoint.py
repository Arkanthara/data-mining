import numpy as np
from lda import LDA


class LDAGD(LDA):
    """
    Implement LDA using GD`.
    """

    def __init__(self, n_components=None):
        super().__init__(n_components=n_components)

    def _calculate_discriminants(self, X, y, iterations=1000):
        """
        This function should compute the linear discriminants and assign them to 
        `self.linear_discriminants`

        Parameters
        ----------
        X : array-like, shape = [num_samples, n_features]
        y : array, shape (num_samples,) containing the target values
        iterations: int, number of gradient descent iterations to run.

        Returns
        -------
        None
        """

