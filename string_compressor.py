def str_comp(instring):

   string = instring
   init = 0
   match = 0
   temp_char = ''
   new_string = ''

   print('String passed in is: ', string)
   for char in string:
     # This is the initial run, so init is set to zero and we copy our first chat to a temp location
     if init == 0:
       temp_char = char
       match += 1

     # When init is 1, we need to compare out temp value to our next value in the list
     if char == temp_char and init != 0:
       # When we have a match, we need to increment our matching counter
       match += 1
       #print('1. here...', char, match)
     elif init != 0:
       # Whenever we have a change in char, we need to copy the temp char into new_string
       new_string = new_string + temp_char + str(match)
       #print('2. now here...', new_string)
       # We reset the temp_char to hold our new char
       temp_char = char
       # Reset counter becasue we have a change
       match = 1
       #print(char, match)

     # At the bottom of the loop, we now want to set init to 1
     init = 1

   compressed_string = new_string + temp_char + str(match)
   if len(compressed_string) < len(string):
      #print("Compressed String : " + compressed_string)
      return compressed_string
   else:
      #print("Original string is less than compressed string : " + string)
      return string
   
   
   return

ret_val = str_comp('aabcccccaaa')
print(ret_val)
ret_val = str_comp('AAaabcccccaaaCC')
print(ret_val)
ret_val = str_comp('abc')
print(ret_val)

# Pass string to the function.
# In the function, setup variables to hold temp values and initial values.
# Initial loop through, set the first char in string to temp and reset init to 1.
# We use init to tell us whether we are going through the loop the first time or not.
# Next time through the loop, compare the first char against the second char.
# If we match then update our match counter.
# If we don't match then set temp to current char
# Continue through the loop until the end
# Finally, check the length of the initial string against the length of the compressed string.
# Return the smaller string back to the caller.
