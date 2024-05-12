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

        # We get the number of classes - 1
        h = self.n_components

        # We get SW and SB matrix
        SW, SB = self.calculate_scatter_matrices(X, y)

        # We initialise w
        w = np.zeros((X.shape[1], h))

        for i in range(h):

            # We compute w[i] for each i in h (w is of size dxh)
            w[:, i] = (np.linalg.inv(SW) @ (self.means[i + 1] - self.means[i]))

        # We replace self.linear_discriminants by the value obtained
        self.linear_discriminants = w
