import urllib.request, json 

# get our data
with urllib.request.urlopen('https://api.ziprecruiter.com/jobs/v1?search=Python&location=Santa%20Monica&api_key=9g9he5ztwqbxs29tpeqpzug72awxnrfr') as url:
    data = json.loads(url.read().decode())
    
# convert our data to a dataframe
import pandas as pd
df = pd.DataFrame(data)

# you can look at how everything is mashed up into a dictionary
# if you do a print(df)

# set up our list_of_lists to hold our dictionary
list_of_lists = []

# loop through the df.jobs column (which is a dict) and append to list_of_lists
for job in df.jobs:
    list_of_lists.append(job)
    
# you print list_of_lists to see that it is a list of dictionaries

# now create our dataframe from our list_of_lists
df1 = pd.DataFrame(list_of_lists)
df1.columns
df1.head()
