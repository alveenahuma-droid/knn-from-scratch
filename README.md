**KNN From Scratch**

A simple implementation of the **K-Nearest Neighbors (KNN)** classification algorithm in Python, built from scratch using only NumPy. This is a beginner-friendly project for learning how KNN works under the hood.

**How It Works**

1. **Fit**: the model stores the training data.
2. **Predict**: for each test point, it:
   - calculates the Euclidean distance to every training point,
   - finds the `k` nearest neighbors,
   - returns the most common class among those neighbors (majority vote).

## Requirements

- Python 3.8+
- NumPy

## Installation

```bash
git clone https://github.com/<your-username>/knn-from-scratch.git
cd knn-from-scratch
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from knn import KNN

X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 8], [8, 7]])
y_train = np.array([0, 0, 0, 1, 1, 1])

knn = KNN(k=3)
knn.fit(X_train, y_train)

X_test = np.array([[2.5, 2.5], [7.5, 7.5]])
print(knn.predict(X_test))   # [0, 1]
```

Or run the built-in example:

```bash
python knn.py
```

## Example Output

```
Test point [2.5 2.5] predicted as class 0
Test point [7.5 7.5] predicted as class 1
```

## Possible Improvements

- Support other distance metrics (Manhattan, Minkowski)
- Add an accuracy/evaluation function
- Vectorize the distance calculation for speed
- Test on real datasets (e.g., Iris)

## License

MIT
