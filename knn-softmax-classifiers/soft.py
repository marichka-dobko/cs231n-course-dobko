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
            batch_intexes = np.random.choice(xrange(num_train), batch_size, replace=False)
            X_batch = X[batch_intexes, :]
            y_batch = y[batch_intexes]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            # perform parameter update
            self.W -= learning_rate * grad.T

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

        y_pred = np.zeros(X.shape[1]).T
        y_pred = self.W.T.dot(X.T)
        y_pred = np.argmax(y_pred, axis=0)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        return self.softmax_loss_vectorized(X_batch, y_batch, reg)

    def softmax_loss_vectorized(self, X, y, reg):
        """
        Softmax loss function, vectorized version.

        Inputs and outputs are the same as softmax_loss_naive.
        """
        # Initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(self.W)
        # print W
        """num_classes = self.W.shape[1]
        num_train = X.shape[0]

        scores = X.dot(self.W)
        scoresExp = np.exp(scores)

        probs = scoresExp / np.sum(scoresExp, axis=1, keepdims=True)
        corect_logprobs = -np.log(probs[range(num_train), y])
        loss = np.sum(corect_logprobs) / num_train

        probs[range(num_train), y] -= 1
        probs /= num_train

        # backpropate the gradient to the parameters (W,b)
        dW = np.dot(X.T, probs)

        loss += 0.5 * reg * np.sum(self.W * self.W)

        dW += reg * self.W
        return loss, dW"""
        X = X.T
        W = self.W.T

        num_classes = W.shape[0]
        num_train = X.shape[1]

        # Compute scores
        f = np.dot(W, X)

        # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
        f -= np.max(f)

        # Loss: L_i = - f(x_i)_{y_i} + log \sum_j e^{f(x_i)_j}
        # Compute vector of stacked correct f-scores: [f(x_1)_{y_1}, ..., f(x_N)_{y_N}]
        # (where N = num_train)
        f_correct = f[y, range(num_train)]
        loss = -np.mean(np.log(np.exp(f_correct) / np.sum(np.exp(f))))

        # Gradient: dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
        p = np.exp(f) / np.sum(np.exp(f), axis=0)
        ind = np.zeros(p.shape)
        ind[y, range(num_train)] = 1
        dW = np.dot((p - ind), X.T)
        dW /= num_train

        # Regularization
        loss += 0.5 * reg * np.sum(W * W)
        dW += reg * W
        return loss, dW