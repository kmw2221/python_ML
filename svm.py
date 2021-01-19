import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)

x = cancer.data  # All of the features
y = cancer.target  # All of the labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print(x_train[:5], y_train[:5])

clf = svm.SVC(kernel="linear", C=2)
# larger C increases the size of your soft margin
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)

print(acc)

# compare to K nearest neighbors

clf2 = KNeighborsClassifier(n_neighbors=9)
clf2.fit(x_train, y_train)
y_pred2 = clf2.predict(x_test)
acc2 = metrics.accuracy_score(y_test, y_pred2)

print(acc2)