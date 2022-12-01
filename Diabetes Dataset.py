"""
Exploring linear regression
"""

"""
Add imports
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection

"""
Shape of the data matrix
"""
X, y = datasets.load_diabetes(return_X_y=True)
print(X.shape)
print(X[0])

"""
Selecting a portion of the dataset to plot by arraning it into a new array
"""
X = X[:, np.newaxis, 2]

"""
Determine a logical split between the numbers in this dataset by splitting the data (X) and the target (y) into test and training sets
"""
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)

"""
Use the linear regression model and train it with the X and y training sets
"""
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

"""
Create a prediction using test data
"""
y_pred = model.predict(X_test)

"""
Display the data in a plot
A scatterplot of all the X and y test data that uses the prediction to draw a line between the model's data groupings
"""
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Scaled BMIs')
plt.ylabel('Disease Progression')
plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
plt.show()
