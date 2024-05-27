import numpy as np
from random import shuffle
from classifier import Classifier


class Logistic(Classifier):
    """A subclass of Classifier that uses the logistic function to classify."""
    def __init__(self, random_seed=0):
        super().__init__('logistic')
        if random_seed:
            np.random.seed(random_seed)



    def loss(self, X, y=None, reg=0):
        """
        Softmax loss function, vectorized version.

        Inputs and outputs are the same as softmax_loss_naive.
        """
        # Initialize the loss and gradient to zero.
        scores = None
        loss = None
        dW = np.zeros_like(self.W)
        num_classes = self.W.shape[1]
        num_train = X.shape[0]

        #scores
        #############################################################################
        # TODO: Compute the scores and store them in scores.                        #
        #############################################################################
        sigmoid = lambda z: 1/(1 + np.exp(-z))

        scores = sigmoid(X @ self.W)

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        if y is None:
            return scores


        # loss
        #############################################################################
        # TODO: Compute the logistic loss and store the loss in loss.               #
        # If you are not careful here, it is easy to run into numeric instability.  #
        # Don't forget the regularization!                                          #
        #############################################################################
        y = y.reshape(-1, 1)
        
        loss = float(- y.T @ np.log(sigmoid(X @ self.W))
                     - (np.ones_like(y.T) - y.T) @ np.log(np.ones_like(y) - sigmoid(X @ self.W))
                     + reg * np.sum(self.W **2))
        loss /= num_train


        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        # grad
        #############################################################################
        # TODO: Compute the gradients and store the gradients in dW.                #
        # Don't forget the regularization!                                          #
        #############################################################################     
        
        dW = X.T @ (sigmoid(X @ self.W) - y) + 2 * reg * self.W
        dW /= num_train
        
        
    
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        return loss, dW

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################

        sigmoid = lambda z: 1 / (1 + np.exp(-z))

        y_pred = sigmoid(X @ self.W)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

