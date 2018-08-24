# project 3
import matplotlib.pyplot as plt

#% matplotlib inline

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()

X = pd.DataFrame(boston.data,
                 columns=boston.feature_names)
y = pd.DataFrame(boston.target,
                 columns=['MEDV'])

print(boston['DESCR'])
print(boston.data.shape)
print(X.describe(include='all'))
print(y.describe(include='all'))
print(X.dtypes)
print(y.dtypes)
X.loc[:, X.isna().any()]
count_nan = len(X) - X.count()
print('COUNT_NAN_X')
print(count_nan)

y.loc[:, y.isna().any()]
count_nan = len(y) - y.count()
print('COUNT_NAN_Y')
print(count_nan)

#feature_cols = X[['RM', 'DIS', 'PTRATIO', 'CRIM']]
feature_RM = X[['RM']]
feature_DIS = X[['DIS']]
feature_PTRATIO = X[['PTRATIO']]
feature_CRIM = X[['CRIM']]

feature_ALL = [feature_RM, feature_DIS, feature_PTRATIO, feature_CRIM]

# step 1
# import the class
from sklearn.linear_model import LinearRegression

# step 2
# instantiate
lr = LinearRegression()

# step 3
# fit
fit_array = []
for feature in feature_ALL:
   fit_array.append(lr.fit(feature, y))
   
# step 4
# predict
predict_array = []
for feature in feature_ALL:
   predict_array.append(lr.score(feature, y))

# print scores
for feature in feature_ALL:
   print(lr.score(feature, y))
   
#print(y)
#for feature in feature_ALL:
#   print(feature.columns)
#   print(lr.coef_)

# score
#print(lr.score(feature_ALL, y))

#import seaborn as sns
#sns.lmplot(x='TEST', y='MEDV', data=feature_ALL[feature], aspect=1.5, scatter_kws={'alpha':0.2});

plt.scatter(feature_RM, y,  color='black')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.savefig("RM" + ".pdf")
plt.scatter(feature_DIS, y,  color='black')
plt.xlabel('DIS')
plt.ylabel('MEDV')
plt.savefig("DIS" + ".pdf")
plt.scatter(feature_PTRATIO, y,  color='black')
plt.xlabel('PTRATIO')
plt.ylabel('MEDV')
plt.savefig("PTRATIO" + ".pdf")
plt.scatter(feature_CRIM, y,  color='black')
plt.xlabel('CRIM')
plt.ylabel('MEDV')
plt.savefig("CRIM" + ".pdf")

# plot our model
#sns.lmplot(x='RM', y='MEDV', data=X, aspect=1.5, scatter_kws={'alpha':0.2});
#for feature in feature_ALL:
#   sns.lmplot(x='TEST', y='MEDV', data=feature_ALL[feature], aspect=1.5, scatter_kws={'alpha':0.2});
   #plt.scatter(feature_ALL[feature], y,  color='black')
   #plt.scatter(feature_RM, y,  color='black')
   #plt.plot(feature_cols[feature], y, color='blue', linewidth=3)
#   plt.xticks(())
#   plt.yticks(())
#   plt.savefig(feature + ".pdf")

from sklearn.model_selection import train_test_split
from sklearn import linear_model

# train/test/split to plot
#X_train, X_test, y_train, y_test = train_test_split(feature_cols, y, random_state=123)

# train/test/split with a ratio of 70% train and 30% test/split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# fit the model on training data
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

# plot and score the model
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.savefig("SCENARIO1.pdf")
print ('Score: ', model.score(X_test, y_test))

# train/test/split with a ratio of 90% train and 10% test/split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# fit the model on training data
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

# plot and score the model
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.savefig("SCENARIO2.pdf")
print ('Score: ', model.score(X_test, y_test))

# k-fold cross validation
#from sklearn.model_selection import KFold
from sklearn import model_selection, metrics

# test k=5 through 10 splits
for split_it in range(5,11):
   kf = model_selection.KFold(n_splits=split_it, shuffle=True)

   mse_values = []
   scores = []
   n = 0

   for train_index, test_index in kf.split(X, y):
      lr = LinearRegression().fit(X.iloc[train_index], y.iloc[train_index])
      mse_values.append(metrics.mean_squared_error(y.iloc[test_index], lr.predict(X.iloc[test_index])))
      scores.append(lr.score(X, y))
      n += 1
	  

   print('Split : {}'.format(split_it))
   print('Mean of MSE for all folds: {}'.format(np.mean(mse_values)))
   print('Mean of R2 for all folds: {}'.format(np.mean(scores)))
   
   