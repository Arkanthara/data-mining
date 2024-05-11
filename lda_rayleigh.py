import numpy as np

from lda import LDA


class LDARayleigh(LDA):
    """
    Implement LDA using the standard method aith Rayleigh quotients.
    """

    def __init__(self, n_components=None):
        super().__init__(n_components=n_components)

    def _calculate_discriminants(self, X, y):
        """
        This function should compute the linear discriminants and assign them to 
        `self.linear_discriminants`, using the rayleigh quotient method with eigenvector decomposition (see slides of the course).

        Parameters
        ----------
        X : array-like, shape = [num_samples, n_features]
        y : array, shape (num_samples,) containing the target values

        Returns
        -------
        None
        """

        SW, SB = self.calculate_scatter_matrices(X, y)

        w = (np.linalg.inv(SW) @ (self.means[1] - self.means[0]))[:, None]

        self.linear_discriminants = w
