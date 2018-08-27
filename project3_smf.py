import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# First, format our data in a DataFrame

df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target
df.head()

# Set up our new statsmodel.formula handling model
import statsmodels.formula.api as smf

# You can easily swap these out to test multiple versions/different formulas
formulas = {
    "case1": "MEDV ~ RM + LSTAT + RAD + TAX + NOX + INDUS + CRIM + ZN - 1", # - 1 = remove intercept
    "case2": "MEDV ~ NOX + RM",
    "case3": "MEDV ~ RAD + TAX",
	"case4": "MEDV ~ RM + LSTAT + RAD + TAX + NOX + INDUS + CRIM - 1", # - 1 = remove intercept
	"case5": "MEDV ~ RM + LSTAT + RAD + TAX + NOX + INDUS - 1", # - 1 = remove intercept
	"case6": "MEDV ~ RM + LSTAT + RAD + TAX + NOX - 1", # - 1 = remove intercept
	"case7": "MEDV ~ RM + LSTAT + RAD + TAX - 1", # - 1 = remove intercept
	"case8": "MEDV ~ RM + LSTAT + RAD - 1", # - 1 = remove intercept
	"case9": "MEDV ~ RM + LSTAT",
	"case10": "MEDV ~ NOX + LSTAT",
}

#model = smf.ols(formula=formulas['case1'], data=df)
#result = model.fit()

#result.summary()

for key, value in formulas.items():
   print(key, value)
   model = smf.ols(formula=formulas[key], data=df)
   result = model.fit()
   result.summary()
      
from sklearn.model_selection import train_test_split
from sklearn import linear_model
y = df['MEDV']

formulas2 = {
    "case1": df[['RM', 'LSTAT', 'RAD', 'TAX', 'NOX', 'INDUS', 'CRIM', 'ZN']],
    "case2": df[['NOX', 'RM']],
    "case3": df[['RAD', 'TAX']],
	"case4": df[['RM', 'LSTAT', 'RAD', 'TAX', 'NOX', 'INDUS', 'CRIM']],
	"case5": df[['RM', 'LSTAT', 'RAD', 'TAX', 'NOX', 'INDUS']],
	"case6": df[['RM', 'LSTAT', 'RAD', 'TAX', 'NOX']],
	"case7": df[['RM', 'LSTAT', 'RAD', 'TAX']],
	"case8": df[['RM', 'LSTAT', 'RAD']],
	"case9": df[['RM', 'LSTAT']],
	"case10": df[['NOX', 'LSTAT']],
}

for key, value in formulas2.items():
   # train/test/split with a ratio of 70% train and 30% test/split
   feature_cols = value
   X = feature_cols
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
   lm = linear_model.LinearRegression()
   model = lm.fit(X_train, y_train)
   predictions = lm.predict(X_test)
   print ('Score for', key, ': ', model.score(X_test, y_test))
   
  
  
# Feature Extraction with RFE
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
#feature_cols = df[['RM', 'LSTAT', 'RAD', 'TAX', 'NOX', 'INDUS', 'CRIM', 'ZN']]
#X = feature_cols
#y = df['MEDV']
# feature extraction
#model = LogisticRegression()
#rfe = RFE(model, 3)
#fit = rfe.fit(X, y)
#print("Num Features: %d") % fit.n_features_
#print("Selected Features: %s") % fit.support_
#print("Feature Ranking: %s") % fit.ranking_