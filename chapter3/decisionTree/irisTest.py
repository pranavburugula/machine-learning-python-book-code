from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
iris_X = iris.data[:, [2, 3]]
iris_y = iris.target
iris_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3, random_state=0)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

print("Training...")
iris_tree.fit(X_train, y_train)
print("Done!")

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
    
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')

def classify_iris():
    iris_predict = iris_tree.predict(X_test)
    print(y_test)
    print(iris_predict)
    plot_decision_regions(X=X_combined, y=y_combined, classifier=iris_tree, test_idx=range(105,150))
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()
    export_graphviz(iris_tree, out_file="tree.dot", feature_names=['petal length', 'petal width'])
    print("Misclassified samples: %d" % (y_test != iris_predict).sum())
    print("Accuracy: %.2f" % accuracy_score(y_test, iris_predict))
    
classify_iris()