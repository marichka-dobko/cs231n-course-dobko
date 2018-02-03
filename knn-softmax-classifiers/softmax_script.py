from __future__ import division
from softmax import *

import sys

X_train = np.load("/home/maria/PycharmProjects/Knn_for_server/X_train.npy")
X_test = np.load("/home/maria/PycharmProjects/Knn_for_server/X_test.npy")
y_train = np.load("/home/maria/PycharmProjects/Knn_for_server/y_train.npy")
y_test = np.load("/home/maria/PycharmProjects/Knn_for_server/y_test.npy")
list_images = np.load("/home/maria/PycharmProjects/Knn_for_server/test_files.npy")

softmax = Softmax()
softmax.train(X_train[:2500], y_train[:2500], learning_rate=3e-8, reg=5e-7,
                      num_iters=150, batch_size=200, verbose=False)

y_train_pred = softmax.predict(X_train[13500:14500])
print y_train_pred
y_t = y_train[13500:14500]
print y_t
s = 0
for i in range(len(y_train_pred)):
    if int(y_t[i]) == int(y_train_pred[i]):
        s += 1
print s
print len(y_train_pred)
acc = s/len(y_train_pred)
print acc

"""myfile = open('results_softmax.txt', 'w')
myfile.write('image,level\n')
s = ''

print len(list_images)
for i in range(len(list_images)):
        s += str(list_images[i])[:-4] + ',' + str(y_test_pred[i]) + "\n"
myfile.write(s)
myfile.close()

with open('my_result.csv', 'w') as csvfile:
    fieldnames = ['image', 'level']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for i in range(len(X_test)):
        writer.writerow({'image': list_images[i][:-4], 'level': predicted_lab[i]})
"""