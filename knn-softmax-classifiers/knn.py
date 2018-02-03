import numpy as np
import random
import os
from scipy.misc import imsave, imread


def load_images(subset):
    path_to_images = os.getcwd() + "/data/" + str(subset)
    labels_file = os.getcwd() + '/data/' + str(subset) + '_labels.txt'
    images_labels = {}
    with open(labels_file, 'r') as f:
        dict_labels = dict([line.strip().split() for line in f.readlines()])

    # List files in this directory
    files = os.listdir(path_to_images)

    # Avoid hidden files
    files = filter(lambda files: not files.startswith('.'), files)
    # files = files[0:1000]

    # Create structure for holding images
    images = np.zeros((len(files), 64, 64, 3), dtype=np.uint8)
    labels = np.zeros(len(files), dtype=np.uint8)

    all_files = []
    for fid, file in enumerate(files):
        if fid % 1000 == 0:
            print fid
        image = imread(path_to_images + '/' + file)
        all_files.append(file)
        if image.shape == (64, 64):
            print file
        images[fid] = image
        labels[fid] = int(dict_labels[file])
    return images, labels, files, all_files


#print "loading train images..."
X_train,y_train,train_files, list_images = load_images('train')
#print "loading test images..."
X_test,y_test,test_files, list_images = load_images('test')



num_validation = 500
num_training = X_train.shape[0] - num_validation

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]



X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


class KNearestNeighbor:
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
    Train the classifier. For k-nearest neighbors this is just
    memorizing the training data.

    Input:
    X - A num_train x dimension array where each row is a training point.
    y - A vector of length num_train, where y[i] is the label for X[i, :]
    """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
    Predict labels for test data using this classifier.

    Input:
    X - A num_test x dimension array where each row is a test point.
    k - The number of nearest neighbors that vote for predicted label
    num_loops - Determines which method to use to compute distances
                between training points and test points.

    Output:
    y - A vector of length num_test, where y[i] is the predicted label for the
        test point X[i, :].
    """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the
    test data.

    Input:
    X - An num_test x dimension array where each row is a test point.

    Output:
    dists - A num_test x num_train array where dists[i, j] is the distance
            between the ith test point and the jth training point.
    """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            for j in xrange(num_train):
                # dists[i, j] = np.linalg.norm(X[i]-self.X_train[j])
                #dists[i, j] = np.sum(np.square(self.X_train[j] - X[i]))
                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
        return dists

    def compute_distances_one_loop(self, X):
        """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            dists[i, :] = np.sum(np.square(self.X_train - X[i, :]), axis=1)
            # dists[i,:] = np.linalg.norm(self.X_train-X[i],axis=1)
        return dists

    def compute_distances_no_loops(self, X):
        """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #ip = X.dot(self.X_train.T)  # inner product
        #XT2 = np.sum(self.X_train ** 2, axis=1)
        #X2 = np.sum(X ** 2, axis=1)
        #dists = -2 * ip + XT2 + X2.reshape(-1, 1)
        dists = np.sqrt(
            (X ** 2).sum(axis=1, keepdims=True) + (self.X_train ** 2).sum(axis=1) - 2 * X.dot(self.X_train.T))


        return dists

    def predict_labels(self, dists, images, k=1):
        """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Input:
    dists - A num_test x num_train array where dists[i, j] gives the distance
            between the ith test point and the jth training point.

    Output:
    y - A vector of length num_test where y[i] is the predicted label for the
        ith test point.
    """
        # print "This is part of the function: predict labels"

        myfile = open('results.txt', 'w')
        myfile.write('image,level\n')
        s = ''

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            neighbors = dists[i, :]
            closest_y = self.y_train[np.argsort(neighbors)[range(k)]]
            y_pred[i] = int(np.argmax(np.bincount(closest_y)))

            s += str(images[i])[:-4] + ',' + str(y_pred[i])[:-2] + "\n"
        myfile.write(s)
        myfile.close()

        return y_pred

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
dists = classifier.compute_distances_two_loops(X_test)
y_test_pred = classifier.predict_labels(dists, list_images, k=1)
