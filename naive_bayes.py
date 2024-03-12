import numpy as np

class Bayes:

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

        if len(X.shape) != 2:
            tmp = []
            try:
                tmp = X.astype('float')
            except:
                values = np.unique(X)
                for i in range(len(X)):
                    print(np.where(values == X[i])[0][0])
                    tmp.append(np.where(values == X[i])[0][0])
            finally:
                return np.array(tmp)


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
        if X.dtype != int or X.dtype != float:
            X = self.string_to_num(X)
        if y.dtype != int or y.dtype != float:
            y = self.string_to_num(y)

        print(X)
        print(y)

        pass
    
    
    def predict(self, X):
        """Predicts the labels for the input data X using the trained
        NaiveBayes object."""
        pass
    
    def prior(self, y):
        """Computes the prior probability of each class given the labels y."""
        pass
    
    def normal_distribution(self, x, mean, std):
        """Computes the probability density function of a normal
        distribution with mean and standard deviation."""
        pass
