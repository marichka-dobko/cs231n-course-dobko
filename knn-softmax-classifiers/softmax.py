import random
import numpy as np
import os
from scipy.misc import imsave, imread
from random import shuffle



class Softmax(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,        # change batch_size default=200
              batch_size=10, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################

            mask = np.random.choice(num_train, batch_size, replace=False)

            X_batch = X[mask]

            y_batch = y[mask]

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            print str(it) + " Iteration"

            # evaluate loss and gradient
            loss, grad = self.softmax_loss_vectorized(X_batch, y_batch, reg)
            loss_history.append(loss)


            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################

            self.W -= learning_rate * grad

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: D x N array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################

        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def softmax_loss_vectorized(self, X, y, reg):
        """
        Softmax loss function, vectorized version.

        Inputs and outputs are the same as softmax_loss_naive.
        """
        # Initialize the loss and gradient to zero.
        W = self.W
        loss = 0.0
        dW = np.zeros_like(W)
        # print W
        num_classes = W.shape[1]
        num_train = X.shape[0]

        scores = X.dot(W)
        scoresExp = np.exp(scores)

        probs = scoresExp / np.sum(scoresExp, axis=1, keepdims=True)
        corect_logprobs = -np.log(probs[range(num_train), y])
        loss = np.sum(corect_logprobs) / num_train

        probs[range(num_train), y] -= 1
        probs /= num_train

        # backpropate the gradient to the parameters (W,b)
        dW = np.dot(X.T, probs)

        loss += 0.5 * reg * np.sum(W * W)

        dW += reg * W
        print dW
        return loss, dW


