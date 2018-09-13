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
   plotname = "output" + str(idx) + ".png"
   
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
   plt.legend({'Mean':mean,'Median':median,'Mode':mode})

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

COLS = ['lead_changes', 'times_tied', 'points_game',
       'field_goals_made', 'field_goals_att', 'field_goals_pct',
       'three_points_made', 'three_points_att', 'three_points_pct',
       'two_points_made', 'two_points_att', 'two_points_pct', 'blocked_att',
       'free_throws_made', 'free_throws_att', 'free_throws_pct',
       'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists',
       'turnovers', 'steals', 'blocks', 'assists_turnover_ratio',
       'personal_fouls', 'ejections', 'foulouts', 'fast_break_pts',
       'second_chance_pts', 'team_turnovers', 'points_off_turnovers',
       'team_rebounds', 'flagrant_fouls', 'player_tech_fouls',
       'team_tech_fouls', 'coach_tech_fouls']
	   
for col in COLS:
   print(COLS.index(col))
   plot_all(col, COLS.index(col))
   


# lets gather the features that we are interested in

# rename population columns
#pop_df.rename(columns={"2010 Census Population": "pop_2010", "Population Estimate, 2011": "pop_2011", "Population Estimate, 2012": "pop_2012", "Population Estimate, 2013": "pop_2013", "Population Estimate, 2014": "pop_2014", "Population Estimate, 2015": "pop_2015", "Population Estimate, 2016": "pop_2016"}, inplace = True)

# for population data, remove commas and convert to int
#for col in pop_df.columns[3:]:
#   pop_df[col] = pop_df[col].str.replace(',', '')
#   pop_df[col] = pop_df[col].astype(str).astype(int)

# get our shape
#pop_df.shape
#pop_df.describe(include='all')

# columns with nan values
#pop_df.loc[:, pop_df.isna().any()]
#count_nan = len(pop_df) - pop_df.count()

# popultaion >= 500000
#pop_df[pop_df.pop_2010 >= 500000]

# open health data
#health_df = pd.read_csv('c:/code/data/health_data.csv')
#health_df.columns

# join health_df to pop_df
#result = pd.concat([pop_df, health_df], axis=1, join='inner')

# keep only the columns we want
#result = result.loc[:,~result.columns.duplicated()]

#result.describe(include='all')

# columns with nan values
#result.loc[:, result.isna().any()]
#count_nan = len(result) - result.count()

# examine column counts
#result[['pop_2010', 'pop_2011', 'pop_2012']].head()

# create a dataframe for analysis
# df = result[['FIPS', 'State', 'County', ]]

# open our data that include population estimates for 2008 and 2009
#pop_df2 = pd.read_csv('c:/code/data/co-est2009-alldata.csv')

# drop columns we don't need
#pop_df2.drop(['SUMLEV', 'REGION', 'DIVISION', 'STATE'], axis=1, inplace=True)

# Only interested in counties, so filter where COUNTY > 0
#pop_df2 = pop_df2[pop_df2.COUNTY > 0]

# now, replace 'County' and 'Parish' in CTYNAME column with blank so we only have county/parish name
#pop_df2['CTYNAME'] = pop_df2['CTYNAME'].str.replace('County', '')
#pop_df2['CTYNAME'] = pop_df2['CTYNAME'].str.replace('Parish', '')

#pop_df2 = pop_df2.CTYNAME.str.replace('County', '')
#pop_df2 = pop_df2.CTYNAME.str.replace('Parish', '')

# check our data
#pop_df2[pop_df2.STNAME == 'Louisiana'].CTYNAME.head(10)

# https://www.census.gov/topics/education.html
# https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml#acsST
# https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ACS_16_1YR_S1501&prodType=table
# HC02_EST_VC18	Percent; Estimate; Percent bachelor's degree or higher
# HC02_EST_VC17	Percent; Estimate; Percent high school graduate or higher
