m = movies['genre'].value_counts()
print(m)
print()
for idx, row in m.iteritems():
	if row > 10:
		#print(idx, row)
		mean_of_genre = movies.loc[movies['genre'] == idx].star_rating.mean()
		print('Mean of genre {} is {}'.format(idx, mean_of_genre))

		
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

#movies = pd.read_csv('c:/code/data/imdb_1000.csv')

movies = pd.read_csv('./data/imdb_1000.csv')
movies.head()

movies.sort_values(by=['duration'], ascending='False')
#print(movies)

#print(movies['content_rating'].value_counts())
movies['content_rating'].replace(["NOT RATED", "APPROVED", "PASSED", "GP"], "UNRATED", inplace=True)
#print(movies.head(100))

movies['content_rating'].replace(["X", "TV-MA"], "NC-17", inplace=True)
movies.head()
#print(movies.loc[movies['content_rating'] == 'NC-17'])

movies[movies.duplicated('title', keep=False)]

movies.columns
movies[['star_rating', 'title']][movies.duplicated('title', keep=False)]

#[movies.groupby(by='genre')].sum()

movies[['genre', 'star_rating']].head()
movies.groupby('genre').agg('count')

#m = movies['genre'].value_counts()
#print(m)
#print()
#for idx, row in m.iteritems():
#	if row > 10:
#		#print(idx, row)
#		mean_of_genre = movies.loc[movies['genre'] == idx].star_rating.mean()
#		print('Mean of genre {} is {}'.format(idx, mean_of_genre))

	  
#movies.loc[movies['genre'] == 'Drama'].mean()

#movies.loc[movies['genre'] == 'Drama'].star_rating.mean()

movies['actor_list1'], movies['actor_list2'], movies['actor_list3'] = movies['actors_list'].str.split(',', 2).str
movies['actor_list1'].describe()

import numpy as np
pd.concat([movies['actors_list']], axis=1, keys=[movies])
cleaned_actor1 = []
cleaned_actor2 = []
cleaned_actor3 = []
merged_list_of_actors = []
for actor1 in movies['actor_list1']:
	actor1 = actor1.replace("[u'", "'").replace('[u"', '"')
	cleaned_actor1.append(actor1)

df_actors1 = pd.DataFrame(np.array(cleaned_actor1).reshape(-1,1), columns = list("a"))
	
for actor2 in movies['actor_list2']:
	actor2 = actor2.replace("u'", "'").replace('u"', '"')
	cleaned_actor2.append(actor2)
	
df_actors2 = pd.DataFrame(np.array(cleaned_actor2).reshape(-1,1), columns = list("a"))

for actor3 in movies['actor_list3']:
	actor3 = actor3.replace("u'", "'").replace("]", "").replace('u"', '"')
	cleaned_actor3.append(actor3)

df_actors3 = pd.DataFrame(np.array(cleaned_actor3).reshape(-1,1), columns = list("a"))

merged_list_of_actors = df_actors1.append([df_actors2, df_actors3])
#print(merged_list_of_actors)
print(merged_list_of_actors.describe())
val_cnt = merged_list_of_actors['a'].value_counts()
#x = merged_list_of_actors.groupby('a').agg('count')
print(val_cnt)
movies[['genre','actor_list1']].groupby(['genre', 'actor_list1']).agg('count')

movies.groupby(['genre', 'actor_list1']).size().sort_values(ascending=False).reset_index(name='count')
movies.groupby(['genre', 'actor_list2']).size().sort_values(ascending=False).reset_index(name='count')
movies.groupby(['genre', 'actor_list3']).size().sort_values(ascending=False).reset_index(name='count')


#print(x)
#print(merged_list_of_actors)
#for y in merged_list_of_actors:
#   print(y)
   
#print()
#merged_list_of_actors.describe()

#df = pd.DataFrame(np.array(merged_list_of_actors).reshape(3,3), columns = list("abc"))


# Create actors_list dataframe
#import ast
#actor_list = movies['actors_list']
#print(actor_list)
#a = [['and', 'by', 'this'], ['do', 'that', 'why']]
#for i in a:
#	for j in i:
#		print(j)
	
#print(actor_list)
#a = ast.literal_eval(actor_list)
#print(*a, sep='\n')
#new_actor_list = []
#for actor in actor_list:
#	print('Here... ', actor)
#	i = actor.split(',')
#	print(j)
#	for j in i:
#		x = j.replace("u'", "'").replace('u"', '"')
#		print(x)
		#new_actor_list.append(j.replace("u'", "'").replace('u"', '"'))
#print(new_actor_list)
	

#print(actor_list)
#actor_list = [actors[1] for actors in actor_list]
#for actor in actor_list:
#	print(actor)

#print(actor_list)
#for actor in actor_list:
#	print(actor)
	#i = actor.split(',')
	#print(i)
# Create another dataframe
#examine_movies = movies[['actors_list', 'genre']]
#for idx, row in examine_movies.iteritems():
#	print(idx, row)
