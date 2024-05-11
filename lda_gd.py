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

        h = 1

        alpha = 0.1

        #w = np.random.random((X.shape[1], h))

        w = np.zeros((X.shape[1], h))
        w[0] = 1

        SW, SB = self.calculate_scatter_matrices(X, y)

        for i in range(iterations):

            J = np.linalg.det(w.T @ SW @ w)/np.linalg.det(w.T @ SB @ w)

            w -= alpha * 2 * J * (
                    SW @ w @ np.linalg.inv(w.T @ SW @ w)
                  - SB @ w @ np.linalg.inv(w.T @ SB @ w))

            for j in range(h):
                w[:, j] /= np.linalg.norm(w[:, j])

        self.linear_discriminants = w
