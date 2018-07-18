dogs = ("beagle", "collie", "healer", "pug")
dogs_characteristics = {}
list_of_characteristics = [1, 25, 'ball']

# Loop through the list and print each value out
for dog in dogs:
   if dog == "healer":
      print(dog)
      # Create the dict object
      dogs_characteristics["breed"] = dog
      dogs_characteristics["age"] = list_of_characteristics[0]
      dogs_characteristics["weight"] = list_of_characteristics[1]
      dogs_characteristics["favorite_toy"] = list_of_characteristics[2]

# When we find a match for the dog type, create a dict obj
# with the key being the dog
print(dogs_characteristics)
