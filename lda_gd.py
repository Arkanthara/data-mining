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

        # We take the number of classes - 1
        h = self.n_components

        # We define the learning rate (I define it to 0.1 because it gives me nice results, but we can modify it if we want)
        alpha = 0.1

        # We initialize w as [Identity | 0 ... 0 ].T
        w = np.eye(X.shape[1], h)

        # We get SW and SB
        SW, SB = self.calculate_scatter_matrices(X, y)

        # We iterate
        for i in range(iterations):
            
            # We compute J
            J = np.linalg.det(w.T @ SW @ w)/ np.linalg.det(w.T @ SB @ w)
            
            # Then we compute the gradient descent
            w -= alpha * 2 * J * (
                    SW @ w @ np.linalg.inv(w.T @ SW @ w)
                  - SB @ w @ np.linalg.inv(w.T @ SB @ w))
            
            # We normalise w
            for j in range(h):
                w[:, j] /= np.linalg.norm(w[:, j])

        # We replace the linear discriminant by the value computed
        self.linear_discriminants = w
