def random_name(names_list):

  hold=[]
  ctr = 0
  tries = 0

  print('Here...')
  noMatch = True

  while noMatch:
     # Populate the hold[] list
     hold=[]
     for i in range(1,4):
        hold.append((random.choice(names_list)))
     # get the first item in the list
     temp = hold[0]
     # Iterate over list and compare temp to hold[1], hold[2]
     for i in hold[1:]:
        if temp != i:
           print(temp, i, hold)
           break
        else:
           # if hold[1] and hold[2] match temp, we have all 3 values match
           ctr =+ ctr + 1
           if ctr == 2:
              print(temp, i, hold, ctr)
              # Set noMatch to False
              noMatch = False

     tries =+ tries + 1
     # Reset ctr to 0
     ctr = 0

  print('it took ', tries, ' tries to get all three matches')
  return True

first_line = 1
listoflists = []

with open('assets/datasets/coffee-preferences.csv', 'r') as f:
  lines = f.readlines()
  for line in lines:
    line = line.replace('\n', '')
    if first_line == 1:
       header = line.split(',')
       header = header[1:]
       first_line = 2
       print('header : {}'.format(header))
    else:
      new_data = []
      data = line.split(',')
      data = data[1:]
      print(data)
      for value in data:
        if not value:
           new_data.append(None)
        else:
           try:
              new_data.append(float(value))
           except:
              new_data.append(value)
      listoflists.append(new_data)

print('listoflists : {}'.format(listoflists))
none_values_dict = {}
count_none_values = 0

# Add to the dictionary
for value in listoflists:
   for none_value in value:
      if type(none_value) == str:
         key = none_value
      elif none_value == None:
         count_none_values =+ count_none_values + 1

   none_values_dict[key] = count_none_values
   count_none_values = 0

print
print(none_values_dict)
print

# Print the dictionary
for k, v in none_values_dict.items():
   print(k, v)

print('Our list of lists is : ', listoflists)
# https://docs.python.org/3/tutorial/datastructures.html
# Take the list of list and transpose it
transposed_listoflists = []
for i in range(10):
    transposed_listoflists.append([row[i] for row in listoflists])

print
print(transposed_listoflists[1:])

# create a list of ratings
ratings = []
for transposed_list in transposed_listoflists[1:]:
   num_ratings = 0
   value = 0
   sum_ratings = 0
   for value in transposed_list:
      if not value:
         None
      elif type(value) == float:
         num_ratings =+ num_ratings + 1
         sum_ratings = sum_ratings + value
         print(value)
   print('Counter = :',  num_ratings, sum_ratings);
   ratings.append(float(sum_ratings)/num_ratings)

print(header[1:])
print(ratings)
# Merge the header list with ratings list to create our dict
dict_of_ratings = dict(zip(header[1:], ratings))
print(dict_of_ratings)

# Create list of names
# From out dict none_values_dict, create a names list
names_list = []

for k, v in none_values_dict.items():
   print(k)
   names_list.append(k)

# print('Names list : ', names_list)

import random
import itertools
matchFound = random_name(names_list)
print(matchFound)
