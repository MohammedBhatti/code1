m = movies['genre'].value_counts()
print(m)
print()
for idx, row in m.iteritems():
	if row > 10:
		#print(idx, row)
		mean_of_genre = movies.loc[movies['genre'] == idx].star_rating.mean()
		print('Mean of genre {} is {}'.format(idx, mean_of_genre))
