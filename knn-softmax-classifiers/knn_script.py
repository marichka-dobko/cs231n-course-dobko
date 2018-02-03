from knn import KNearestNeighbor
import time
start_time = time.time()


classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
dists = classifier.compute_distances_two_loops(X_test)

print("--- %s seconds ---" % (time.time() - start_time))
y_test_pred = classifier.predict_labels(dists, list_images, k=1)