import numpy as np
import matplotlib.pyplot as plt


class LDA:
    """
    Base class for an LDA classifier.
    """

    def __init__(self, n_components=None):

        self.n_components = n_components

        # Shape will be (num_features, n_components)
        self.linear_discriminants = None

        # This vector should be filled with the mean vectors for each class in the dataset.
        # Each vector is of shape (num_feature,)
        # and each element if the mean value of the corresponding feature in the dataset.
        # One filled, for the binary classification task, this list will thus hold two elements (one for class), each one with shape (num_features,)
        self.means = []

        self.mean_overall = 0.

    def train(self, X, y):
        """Train the model.

        This function should call the appropriate method 
        and set self `self.discriminants`.

        
        Parameters
        ----------
        X : array-like, shape (num_samples, n_features)
            Training vectors, where num_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (num_samples,)
            Target values.

        Returns
        -------
        None
        """

        # We compute the discriminant. So the field self.linear_discriminant is completed
        self._calculate_discriminants(X, y)


    def calculate_scatter_matrices(self, X, y):
        """

        This function should compute and return the within-class and between-class scatter matrices.

        Suggestion: fill `self.means` in this method.

        Parameters
        ----------
        X : array-like, shape = [num_samples, n_features]
        y : array, shape (num_samples,) containing the target values

        Returns
        -------

        Tuple (SW, SB), where
            SW : array-like, shape: [n_features, n_features]
            SB : array-like, shape: [n_features, n_features]
        """

        # We define h = number of classes - 1
        h = self.n_components

        # We get all the classes and we sort then (to be sure to access later to mean of right class)
        classes = np.sort(np.unique(y))

        # We initialise matrix SW and SB
        SW = np.zeros((X.shape[1], X.shape[1]))
        SB = np.zeros((X.shape[1], X.shape[1]))

        # Here we define a variable to detect if we have already compute the means
        add_means = False
        if len(self.means) == 0:
            add_means = True
            # Add the overall mean
            self.mean_overall = np.mean(X, axis = 0)

        # We iterate on each classes (so on h + 1)
        for i in range(len(classes)):

            # This give us only the datas where the label is equal to the current class
            X_ci = X[y == classes[i]]

            # If we didn't compute the means, we compute and add the means to self.means
            if add_means:
                self.means.append(np.mean(X_ci, axis=0))

            # Then, we compute SW
            SW += (X_ci - self.means[i]).T @ (X_ci - self.means[i])
            
            # And SB
            SB += X_ci.shape[0] * (self.means[i] - self.mean_overall).reshape(-1, 1) @ (self.means[i] - self.mean_overall).reshape(1, -1)
        
        return SW, SB

    def _calculate_discriminants(self, X, y):
        """

        This function should compute the linear discriminants and assign them to 
        `self.linear_discriminants`.

        The function is not implemented in the `LDA` base class. 
        Different implementations will be provided for each required approach specified in the TP.

        Parameters
        ----------
        X : array-like, shape = [num_samples, n_features]
        y : array, shape (num_samples,) containing the target values

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def transform(self, X):
        """

        Parameters
        ----------
        X : array-like, shape = [num_samples, n_features]

        Returns
        -------
        predictions : array, shape = [self.n_components]
            Projections of input samples using the linear discriminants in `self.linear_discriminants`.
        """

        # We make a projection transformation thanks to the linear_discriminant computed before
        return X @ self.linear_discriminants

    def predict(self, X):
        """
        Returns predictions for binary classification task.

        For the decision boundary, we use a simple heuristic:
        The threshold value is computed by averaging the mean value for each class 
        that you should have computed in `self.means`.

        For binary classificaiton, given class means M1 and M2, compute M = (M1+M2)/2.
        Then, this value is projectd onto the linear discriminants to give the threshold value.

        This threshold should be used to perform the classification decision. 
        That is, for a test point projected onto the linear discriminants, with projection x', 
        x'>threshold gives us the positive class, while x'<= threshold holds the negative class.

        Parameters
        ----------
        X : array-like, shape = [num_samples, n_features]

        Returns
        -------
        predictions : array, shape = [num_samples]
            Predicted target values for X
        """
        assert self.linear_discriminants is not None

        # There should only be 2 mean values (one for each class)
        # as this is binary classification.
        assert len(self.means) == 2

        # We compute M used to obtain threshold
        M = (self.means[0] + self.means[1]) / 2

        # We compute threshold
        threshold = self.transform(M)
        
        # I add [:, 0] to convert 2D vector of size num_samples, 1 to 1D array
        predictions = self.transform(X)[:, 0]

        # We make predictions. Note that here, we have inverted the order of the prediction
        predictions[predictions > threshold] = 1
        predictions[predictions <= threshold] = 0

        return predictions

    def plot_1d(self, X, y, title=None):
        """ This function plots the projected datapoints to a single line.
        This should be used for datasets with two clases, where `self.n_components == 1`.

        Note: plot the 1D projectios on a single line. You can choose the line where y=0
        to plot the points on. Assign a different color to each class.
        """

        assert self.n_components == 1

        # We get all classes
        values = np.unique(y)

        # Plot figure
        plt.figure()
        plt.title(title)

        # We iterate on each class
        for i in values:

            # We take only the datas which are in the class
            X_i = X[y == i]

            # We project this datas thanks to the linear discriminant
            projection = self.transform(X_i)[:, 0]

            # We plot the datas projected
            plt.plot(projection, np.zeros_like(projection), label="class " + str(i))

        # This is just used to plot the threshold
        M = (self.means[0] + self.means[1]) / 2

        # We compute threshold thanks to M
        threshold = self.transform(M)

        # We plot threshold
        plt.plot(threshold, 0, 'ro', label="threshold")

        plt.legend()
        plt.show()



    def plot_2d(self, X, y, title=None):
        """ Plot the dataset X and the corresponding labels y in 2D using the LDA
        transformation.
        Assign a different color to each class.
        """

        assert self.n_components == 2

        # We take all the classes
        values = np.unique(y)

        plt.figure()

        plt.title(title)

        # For each classes
        for i in values:

            # We take the datas which are in this class
            X_i = X[y == i]

            # We project the data thanks to the linear discriminant
            projection = self.transform(X_i)

            # We print our data projected
            plt.plot(projection[:, 0], projection[:, 1], 'o', label="class " + str(i))

        plt.legend()
        plt.show()


