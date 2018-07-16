#!/usr/bin/python

string = "aabcccccaaa"
init = 0
match = 0
temp_char = ''
new_string = ''

print('String we have is: aabcccccaaa')
for char in string:
  # This is the initial run, so init is set to zero and we copy our first chat to a temp location
  if init == 0:
    temp_char = char
    match += 1

  # When init is 1, we need to compare out temp value to our next value in the list
  if char == temp_char and init <> 0:
    # When we have a match, we need to increment our matching counter
    match += 1
    print('1. here...', char, match)
  elif init <> 0:
    # Whenever we have a change in char, we need to copy the temp char into new_string
    new_string = new_string + temp_char + str(match)
    print('2. now here...', new_string)
    # We reset the temp_char to hold our new char
    temp_char = char
    # Reset counter becasue we have a change
    match = 1
    print(char, match)

  # At the bottom of the loop, we now want to set init to 1
  init = 1

print
print("String we got is: " + new_string + temp_char + str(match))
print
print('Result we want is: a2b1c5a3')
