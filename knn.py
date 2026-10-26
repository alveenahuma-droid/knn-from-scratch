import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate distances
            distances = [np.sqrt(np.sum((x - x_train) ** 2))
                         for x_train in self.X_train]

            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]

            # Majority vote
            most_common = Counter(k_labels).most_common(1)
            predictions.append(most_common[0][0])
        return predictions


if __name__ == "__main__":
    X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 8], [8, 7]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    X_test = np.array([[2.5, 2.5], [7.5, 7.5]])
    predictions = knn.predict(X_test)

    print(f"Test point {X_test[0]} predicted as class {predictions[0]}")
    print(f"Test point {X_test[1]} predicted as class {predictions[1]}")
