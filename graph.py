from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from utils import train_test_split, compute_accuracy
import numpy as np


dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
X_train, y_train, X_test, y_test = train_test_split(X, y, 0.3, normalize=True)

# import classifier 
from lda_gd  import LDAGD

iterations = np.arange(1, 1000, 10).astype(int)

alpha = np.linspace(0.1, 1., 100)

lda_clf = LDAGD(n_components=1)
lda_clf._calculate_discriminants(X_train, y_train, alpha=1)
y_pred_train = lda_clf.predict(X_train)
y_pred_test = lda_clf.predict(X_test)
train_accuracy = compute_accuracy(y_train, y_pred_train)
test_accuracy = compute_accuracy(y_test, y_pred_test)

print("Learning rate = 1")
print(f"LDA_GD train accuracy: {train_accuracy}")
print(f"LDA_GD test accuracy: {test_accuracy}")

lda_clf = LDAGD(n_components=1)
lda_clf._calculate_discriminants(X_train, y_train, alpha=0.1)
y_pred_train = lda_clf.predict(X_train)
y_pred_test = lda_clf.predict(X_test)
train_accuracy = compute_accuracy(y_train, y_pred_train)
test_accuracy = compute_accuracy(y_test, y_pred_test)

print("Learning rate = 0.1")
print(f"LDA_GD train accuracy: {train_accuracy}")
print(f"LDA_GD test accuracy: {test_accuracy}")


train_accuracy = []
test_accuracy = []

for i in iterations:
    lda_clf = LDAGD(n_components=1)
    lda_clf._calculate_discriminants(X_train, y_train, i)
    y_pred_train = lda_clf.predict(X_train)
    y_pred_test = lda_clf.predict(X_test)
    train_accuracy.append(compute_accuracy(y_train, y_pred_train))
    test_accuracy.append(compute_accuracy(y_test, y_pred_test))


plt.figure()
plt.title("Accuracy for differents iterations in gradient descent")
plt.plot(iterations, train_accuracy, 'o', label="train accuracy")
plt.plot(iterations, test_accuracy, 'x', label="test accuracy")
plt.xlabel("number of iterations")
plt.legend()
plt.show()

