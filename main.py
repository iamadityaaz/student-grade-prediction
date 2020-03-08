import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")

print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

print(data)
predict = "G3"

# gonna return newdata frame, array, that doesnt have G3 in it
X = np.array(data.drop([predict], 1))
# we only care about G3 value here
Y = np.array(data[predict])

# diving it into for parts ,aand storing our test data that is 10% - 0.1
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, Y, test_size=0.1)

linear = linear_model.LinearRegression()

'''
best = 0
for _ in range(1000):
    # diving it into for parts ,aand storing our test data that is 10% - 0.1
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.1)

    linear = linear_model.LinearRegression()
    # feed data to find best fit line, and stoire it in linear
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickel", "wb") as f:
            pickle.dump(linear, f)'''


pickle_in = open("studentmodel.pickel", "rb")
linear = pickle.load(pickle_in)

print("Co : ", linear.coef_)
print("Intercept : ", linear.intercept_)


predictions = linear.predict(x_test)


for x in range(len(predictions)):
    print("Prediction : {p}, \t x_test : {x}, \t y_test : {y}".format(
        p=predictions[x], x=x_test[x], y=y_test[x]))

p = "G3"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()
