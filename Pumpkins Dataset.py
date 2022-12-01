"""
data exploration
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# check for missing dates
# view the first five rows
pumpkins = pd.read_csv('./US-pumpkins.csv')

pumpkins.head()

# add a filter for pumpkins by the bushel
pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]

# check if there is missing data
pumpkins.isnull().sum()

# dropping several columns, using drop(), keeping only the columns needed
new_columns = ['Package', 'Variety', 'City Name', 'Month', 'Low Price', 'High Price', 'Date']
pumpkins = pumpkins.drop([c for c in pumpkins.columns if c not in new_columns], axis=1)

# calculate the average price of a pumpkin
price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

month = pd.DatetimeIndex(pumpkins['Date']).month
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)

# copy converted data into a new Pandas dataframe
new_pumpkins = pd.DataFrame(
    {'Month': month, 
     'DayOfYear' : day_of_year, 
     'Variety': pumpkins['Variety'], 
     'City': pumpkins['City Name'], 
     'Package': pumpkins['Package'], 
     'Low Price': pumpkins['Low Price'],
     'High Price': pumpkins['High Price'], 
     'Price': price})

# normalize the pricing so that you show the pricing per bushel
new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/1.1
new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price*2

new_pumpkins.head()

# plot the data as a box
price = new_pumpkins.Price
month = new_pumpkins.Month
plt.scatter(price, month)
plt.show()

# create a grouped bar chart
new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
plt.ylabel("Pumpkin Price")

# plot each pumpkin category using a different color
ax = None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)

# see what effect the date has on the price
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 

# remove missing values to clean data
pie_pumpkins.dropna(inplace = True)
pie_pumpkins.info()

"""
linear regression
"""
# to train the Linear Regression model the Scikit-learn library is used
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# separating input values (features) and the expected output (label) into separate numpy arrays
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']

# split the data into train and test datasets, so that we can validate our model after training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# training linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

# coefficients of the regression and y-intercept
lin_reg.coef_
lin_reg.intercept_

# see how accurate the model is by predicting prices on a test dataset and then measure how close predictions are to the expected values
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

# coefficient of determination
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)

# plot the test data together with the regression line 
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)

"""
polynomial regression
"""
# create a pipeline that adds polynomial features to model and then trains the regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)

"""
categorical features
"""
# one-hot encode the variety column
pd.get_dummies(new_pumpkins['Variety'])

# set up training data for polynomial regreession
# add more categorical features and numeric features to improve prediction accuracy
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# make train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# setup and train the pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# predict results for test data
pred = pipeline.predict(X_test)

# calculate MSE and determination
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)

"""
logistic regression
"""
# clean the data and dropping null values and selecting only some of the columns
from sklearn.preprocessing import LabelEncoder

new_columns = ['Color','Origin','Item Size','Variety','City Name','Package']

new_pumpkins = pumpkins.drop([c for c in pumpkins.columns if c not in new_columns], axis=1)

new_pumpkins.dropna(inplace=True)

new_pumpkins = new_pumpkins.apply(LabelEncoder().fit_transform)

new_pumpkins.info

# side-by-side grid
import seaborn as sns

g = sns.PairGrid(new_pumpkins)
g.map(sns.scatterplot)

# swarm plot to show the distribution of values
sns.swarmplot(x="Color", y="Item Size", data=new_pumpkins)

# violin plot is useful to visualize the way that data in the two categories is distributed
sns.catplot(x="Color", y="Item Size", kind="violin", data=new_pumpkins)

# binary classification model 
from sklearn.model_selection import train_test_split

X = new_pumpkins['Origin','Item Size','Variety','City Name','Package']
y = new_pumpkins['Color']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))

# confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)

# visualize Receiving Operating Characteristic (ROC) curve
from sklearn.metrics import roc_curve, roc_auc_score

y_scores = model.predict_proba(X_test)
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
sns.lineplot([0, 1], [0, 1])
sns.lineplot(fpr, tpr)

# compute the actual Area Under the Curve (AUC)
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
