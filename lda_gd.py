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

        # We define the learning rate
        alpha = 1

        # We initialize w as [Identity | 0 ... 0 ].T
        w = np.eye(X.shape[1], h)

        # We define a lambda function to avoid division by 0
        divide = lambda p, q: p/q if q != 0 else 0

        # We get SW and SB
        SW, SB = self.calculate_scatter_matrices(X, y)

        # We iterate
        for i in range(iterations):
            
            # We compute J
            J = divide(np.linalg.det(w.T @ SW @ w), np.linalg.det(w.T @ SB @ w))
            
            # We verify if J = 0.
            # If J = 0, it means that for instante, np.linalg.det(w.T @ SB @ w) = 0 so that w.T @ SB @ w is not invertible
            # I use this because else, I have numpy exceptions...
            # Then we compute the gradient descent
            if J != 0:
                w -= alpha * 2 * J * (
                    SW @ w @ np.linalg.inv(w.T @ SW @ w)
                  - SB @ w @ np.linalg.inv(w.T @ SB @ w))
            else: 
                w -= 0
            
            # We normalise w
            for j in range(h):
                w[:, j] /= np.linalg.norm(w[:, j])

        # We replace the linear discriminant by the value computed
        self.linear_discriminants = w
