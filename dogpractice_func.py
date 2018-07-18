def convert_to_dict(name_of_dict, inlist):

  for value in inlist:
     s = value.split(':')
     name_of_dict.update({s[0]:s[1]})

  return

dog1 = {}
dog2 = {}
dog3 = {}
dog4 = {}
list1 = ["breed:beagle", "age:1.5", "weight:20", "favorite_toy:stick"]
list2 = ["breed:collie", "age:2", "weight:35", "favorite_toy:ball and stick"]
list3 = ["breed:healer", "age:.5", "weight:15", "favorite_toy:rubber duck"]
list4 = ["breed:pug", "age:4", "weight:40", "favorite_toy:bone"]
convert_to_dict(dog1, list1)
convert_to_dict(dog2, list2)
convert_to_dict(dog3, list3)
convert_to_dict(dog4, list4)
print
print(dog1)
print(dog2)
print(dog3)
print(dog4)
