import numpy as np

class Bayes:

    parameters = []
    prior_y = {}

    def __init__(self):
        """Initializes the NaiveBayes object."""
        print("Hello")
        pass

    def string_to_num(self, X: np.ndarray) -> np.ndarray:
        """
        Convert an array of string to an array of numbers
        
        Parameters
        ----------
        X: np.ndarray
            The array to convert
        
        Return value
        ------------
        An array of number
        """

        # Case vector
        if len(X.shape) != 2:
            tmp = []
            try:
                tmp = X.astype('float')
            except:
                values = np.unique(X)
                for i in range(len(X)):
                    tmp.append(np.where(values == X[i])[0][0])
            finally:
                return np.array(tmp)

        # Case matrix
        result = []
        for i in range(X.shape[1]):
            tmp = []
            try:
                tmp = X[:, i].astype('float')
            except:
                values = np.unique(X[:, i])
                for j in range(X.shape[0]):
                    tmp.append(np.where(values == X[j, i])[0][0])
            finally:
                result.append(tmp)
        return np.array(result).T

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Trains the NaiveBayes object on the input data X and labels y."""

        X_work = X.copy()
        y_work = y.copy()
        if X.dtype != int or X.dtype != float:
            X_work = self.string_to_num(X)
        if y.dtype != int or y.dtype != float:
            y_work = self.string_to_num(y)

        self.prior_y = self.prior(y_work)
        values = np.unique(y_work)
        n = len(y_work)
        for i in range(len(values)):
            newData = X_work[y_work == values[i]]
            tmp = []
            for j in range(newData.shape[1]):
                newvalues, count = np.unique(newData[:, j], return_counts=True)

                # Case bernouli model
                if len(newvalues) == 2:
                    tmp.append([])
                    for k in len(newvalues):
                        tmp[-1].append([newvalues[i], count[i] / len(newData)])
                # Case Gaussian
                else:
                    tmp.append([np.mean(newData[:, j]), np.var(newData[:, j])])
            self.parameters.append([values[i], tmp])

        pass

    def gaussian(self, x: float, params: list) -> float:

        assert len(params) == 2, "must give mean and variance in params !"

        return 1 / (float(params[1]) * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x - float(params[0])) / float(params[1]))**2)

    def predict_simple(self, x: np.ndarray) -> int:
        """
        Predict a label for the given parameters

        Parameters
        ----------
        x: np.ndarray
            the given parameters

        Return value
        ------------
        The predicted value
        """

        assert len(x.shape) == 1
        
        predict = 0
        maximum = 0
        for i in range(len(self.parameters)):
            value = 1
            for j in range(len(self.parameters[i][1])):
                # case bernouli
                try:
                    len(self.parameters[i][1][j][0])
                    if self.parameters[i][1][j][0][0] == x[j]:
                        value *= self.parameters[i][1][j][0][1]
                    else:
                        value *= self.parameters[i][1][j][1][1]
                # Case gaussian
                except:
                    value *= self.gaussian(float(x[j]), self.parameters[i][1][j])
            if value * self.prior_y[self.parameters[i][0]] > maximum:
                predict = self.parameters[i][0]
                maximum = value
        return predict
    
    def predict(self, X):
        """Predicts the labels for the input data X using the trained
        NaiveBayes object."""
        
        predict = []
        for i in X:
            predict.append(self.predict_simple(i))
        return np.array(predict)
    
    def prior(self, y):
        """Computes the prior probability of each class given the labels y."""
        values, counts = np.unique(y, return_counts=True)
        result = {}
        for i in range(len(values)):
            result[values[i]] = counts[i] / len(y)
        return result
    
    def normal_distribution(self, x, mean, std):
        """Computes the probability density function of a normal
        distribution with mean and standard deviation."""
        pass
