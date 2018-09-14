import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
plt.style.use('fivethirtyeight')
#%matplotlib inline

df = pd.read_csv('c:/code/data/mbb_teams_games_sr.csv')

# drop columns we don't need
df.drop(['game_id', 'status', 'coverage', 'logo_large', 'logo_medium', 'logo_small', 'possession_arrow', 'venue_id', 'team_id', 'league_id', 'conf_id', 'conf_name', 'division_name', 'opp_id', 'opp_league_id', 'opp_league_name', 'opp_conf_id', 'opp_conf_name', 'opp_division_id', 'opp_division_name', 'opp_logo_large', 'opp_logo_medium', 'opp_logo_small',], axis=1, inplace=True)

# do some analysis
print(df.head())
print(df.dtypes)
print(df.describe())
print(df.shape)
print(df.describe(include='all'))
print(df.columns)
#columns with nan values
df.loc[:, df.isna().any()]
count_nan = len(df) - df.count()
print('COUNT_NAN')
print(count_nan)

# lets plot wins/losses home vs away
df[['win','season']]
# groupby league_alias
#print(df[['league_alias']].groupby(['league_alias']).head())
#print(df[['league_alias']].groupby(['league_alias']).league_alias.value_counts())
print(df.groupby(['league_alias']).league_alias.value_counts())
print(df.groupby(['conf_alias']).conf_alias.value_counts())
#print(df.groupby(['status']).head())
#print(df.groupby(['season']).head())

# do we have anything other than 'closed' in the 'status' column?
# print(df[df[['status']]['status'] != 'closed'].head())

# covariance
print(df.cov())

# Calculate the correlation matrix using the default method.
print(df.corr())

sns.set_palette("coolwarm", 7)
sns_heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1)
fig = sns_heatmap.get_figure()
fig.savefig("output.png") 

df.drop(columns=['opp_fast_break_pts', 'opp_second_chance_pts', 'opp_team_turnovers', 'opp_fast_break_pts', 'opp_second_chance_pts', 'opp_team_turnovers',
                 'opp_player_tech_fouls', 'opp_team_tech_fouls', 'opp_coach_tech_fouls', 'created'], inplace=True)
df.drop(columns=['opp_steals', 'opp_blocks', 'opp_assists_turnover_ratio',
       'opp_personal_fouls', 'opp_ejections', 'opp_foulouts', 'opp_points',
       'opp_points_off_turnovers', 'opp_team_rebounds', 'opp_flagrant_fouls'], inplace=True)
df.drop(columns=['opp_points_game', 'opp_minutes', 'opp_field_goals_made',
       'opp_field_goals_att', 'opp_field_goals_pct', 'opp_three_points_made',
       'opp_three_points_att', 'opp_three_points_pct', 'opp_two_points_made',
       'opp_two_points_att', 'opp_two_points_pct', 'opp_blocked_att',
       'opp_free_throws_made', 'opp_free_throws_att', 'opp_free_throws_pct',
       'opp_offensive_rebounds', 'opp_defensive_rebounds', 'opp_rebounds',
       'opp_assists', 'opp_turnovers'], inplace=True)
df.drop(columns=['season', 'neutral_site', 'scheduled_date', 'gametime',
       'conference_game', 'tournament', 'tournament_type', 'tournament_round',
       'tournament_game_no', 'attendance', 'periods', 'venue_city', 'venue_state', 'venue_address', 'venue_zip',
       'venue_country', 'venue_name', 'venue_capacity', 'name',
       'market', 'alias', 'league_name', 'league_alias', 'conf_alias',
       'division_id', 'division_alias', 'opp_name', 'opp_market', 'opp_alias',
       'opp_league_alias', 'opp_conf_alias', 'opp_division_alias'], inplace=True)

# get home wins query
df_home_wins = df.loc[(df['win'] == True) & (df['home_team'] == True)].copy()

df_home_wins[['points','points_game']].head(5)
df_home_wins.describe(include='all')

def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns
		

missing_values_table(df_home_wins)
# points and points_game are the same, but points has missing values, so drop points
df_home_wins.drop(columns=['points'], inplace=True)
df_home_wins[['points_game']].head(100)

# plot the distribution of data
def plot_all(col, idx):
   sns_displot=''
   fig1=''
   plotname = col + str(idx) + ".png"
   
   # get mean, median, mode, and standard deviation
   mean1=df_home_wins[col].dropna().mean()
   median1=df_home_wins[col].dropna().median()
   mode1=df_home_wins[col].dropna().mode().get_values()[0]
   sd1=df_home_wins[col].dropna().std()

   # plot attributes
   fig1, ax = plt.subplots(sharex=True)
   fig1.set_size_inches(11.7, 8.27)

   sns_displot = sns.distplot(df_home_wins[col].dropna(), ax=ax)
   sns_displot = ax.axvline(mean1, color='r', linestyle='--')
   sns_displot = ax.axvline(median1, color='g', linestyle='-')
   sns_displot = ax.axvline(mode1, color='b', linestyle='-')

   # set the legend
   plt.legend({'Mean':mean1,'Median':median1,'Mode':mode1})

   fig1 = sns_displot.get_figure()
   fig1.savefig(plotname)
   
   # figure out which is the best metric to use to fill missing values
   dist1 = abs(mean1 - sd1)
   dist2 = abs(mean1 - sd1)
   dist3 = abs(mode1 - sd1)
   fill_value = min(dist1, dist2, dist3)
   print('the fill value should be : ', fill_value)
   
   # fill the df with fill_value
   #df_home_wins[col] = df_home_wins[col].replace(np.nan, fill_value, inplace=True)

COLS = COLS = ['lead_changes', 'times_tied', 'points_game', 'field_goals_made', 'field_goals_att',
				'three_points_made', 'three_points_att', 'two_points_made', 'two_points_att', 'blocked_att',
				'free_throws_made', 'free_throws_att', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks',
				'assists_turnover_ratio', 'personal_fouls', 'ejections', 'foulouts', 'fast_break_pts', 'second_chance_pts',
				'team_turnovers', 'points_off_turnovers', 'team_rebounds']
  
	   
for col in COLS:
   print(COLS.index(col))
   plot_all(col, COLS.index(col))
   

# fill in NaNs with mean
df_home_wins.fillna(df_home_wins.mean(), inplace=True)

# Create X and y.
# too co-linear 
feature_cols = ['lead_changes', 'times_tied', 'field_goals_made', 'field_goals_att',
'three_points_made', 'three_points_att', 'two_points_made', 'two_points_att', 'blocked_att',
'free_throws_made', 'free_throws_att', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks',
'assists_turnover_ratio', 'personal_fouls', 'ejections', 'foulouts', 'fast_break_pts', 'second_chance_pts',
'team_turnovers', 'points_off_turnovers', 'team_rebounds']

# this one seems right
feature_cols = ['lead_changes', 'times_tied', 
'three_points_made', 'two_points_made', 'blocked_att',
'free_throws_made', 'free_throws_att', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks',
'assists_turnover_ratio', 'personal_fouls', 'ejections', 'foulouts', 'fast_break_pts', 'second_chance_pts',
'team_turnovers', 'points_off_turnovers', 'team_rebounds']

X = df_home_wins[feature_cols]
y = df_home_wins.points_game

# check the heatmap for the above
sns.set_palette("coolwarm", 7)
sns_heatmap = sns.heatmap(X.corr(), vmin=-1, vmax=1)
fig.set_size_inches(11.7, 8.27)
fig = sns_heatmap.get_figure()
fig.savefig("X_HeatMap.png", dpi=fig.dpi) 

#TTS

# Import, instantiate, fit.
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
# instantiate a lr model
linreg = LinearRegression()

# train/test/split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit the model
linreg.fit(X_train, y_train)

# get the predictions
predictions = linreg.predict(X_test)
# Print the coefficients.
print(linreg.intercept_)
print(linreg.coef_)

dict(zip(linreg.coef_,X.columns))
# view the predictions
print(predictions)

## The line / model
plt = ''
import matplotlib.pyplot as plt
#fig = plt.figure(figsize=(3, 6))
#fig = plt.figure(figsize=(12, 12))
fig = plt.figure()
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
fig.savefig("fig2.png")
plt.close()

# and check for accuracy
print("Score:", linreg.score(X_test, y_test))

# can we improve on this using KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Find test accuracy for all values of K between 1 and 100 (inclusive).
# check accuracy
#k_range = list(range(1, 16000, 500))
#training_error = []
#testing_error = []

# Find test accuracy for all values of K between 1 and 100 (inclusive).
#for k in k_range:
#    # Instantiate the model with the current K value.
#    knn = KNeighborsClassifier(n_neighbors=k)
#    knn.fit(X_train, y_train)
#    # print the accuracy
#    y_pred_class = knn.predict(X_test)
#    print('K value versus Accuracy: {}'.format(k), format((metrics.accuracy_score(y_test, y_pred_class))))	


#scores = []
#for k in k_range:
#    knn = KNeighborsClassifier(n_neighbors=k)
#    knn.fit(X,y)
#    pred = knn.predict(X)
#    score = float(sum(pred == y)) / len(y)
#    scores.append([k, score])

#data = df_home_wins.DataFrame(scores,columns=['k','score'])
#data.plot.line(x='k',y='score');

# error rates for different k values
# https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt

rmse_val = [] #to store rmse values for different k
for K in range(101):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error) 
	

# this is our feature set
feature_cols = ['lead_changes', 'times_tied', 
'three_points_made', 'two_points_made', 'blocked_att',
'free_throws_made', 'free_throws_att', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks',
'assists_turnover_ratio', 'personal_fouls', 'ejections', 'foulouts', 'fast_break_pts', 'second_chance_pts',
'team_turnovers', 'points_off_turnovers', 'team_rebounds']

#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
#curve.plot()
fig = curve.plot().get_figure()
fig.savefig('curve.png')

# best k to use is around 7.25
# instantiate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model on the training set (using K=7)
knn = KNeighborsClassifier(n_neighbors=7)
# fit the model
knn.fit(X_train, y_train)

# Test the model on the testing set and check the accuracy
y_pred_class = knn.predict(X_test)
print((metrics.accuracy_score(y_test, y_pred_class)))

# get the predictions
predictions = linreg.predict(X_test)
# Print the coefficients.
print(linreg.intercept_)
print(linreg.coef_)

dict(zip(linreg.coef_,X.columns))
# view the predictions
print(predictions)

## The line / model
plt = ''
import matplotlib.pyplot as plt
#fig = plt.figure(figsize=(3, 6))
#fig = plt.figure(figsize=(12, 12))
fig = plt.figure()
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
fig.savefig("fig2.png")
plt.close()

# and check for accuracy
print("Score:", linreg.score(X_test, y_test))



#RandomForestRegressor
# Finding the important features
from sklearn.ensemble import RandomForestClassifier

