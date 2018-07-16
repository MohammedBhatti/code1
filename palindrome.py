#!/usr/bin/python

def check_palindrome(product):

  # Convert our int into str
  prod_2_str = str(product)
  # Get it's length
  len_of_prod = len(prod_2_str)
  # Get the integer part of the division
  int_div = len(prod_2_str)//2

  # Slice the front from 0 to the middle
  front = prod_2_str[0:int_div]
  # Slice the back
  back = prod_2_str[-int_div:]
  # Reverse the back so that we can compare front and back
  reverse = back[::-1]

  # If front and back are the same, then we have a palindrome
  if (front == reverse):
    print(product, front, back, reverse)
    # Retrurn the palindrome number
    return product


## MAIN
# Define a list to hold number between 100 and 1000
nums = []

# This var holds the product of the largest palindrome number
largest = 0

# Populate our list
for i in range(100,1000):
  nums.append(i)

# For each value in the outer loop, multiply each value in the inner loop
# Outer loop
for outer in nums[:]:
  # Inner loop
  for inner in nums[:]:
    product = outer * inner
    largest_ret = check_palindrome(product)
    # Check which is the largest product that is a palindrome
    if(largest_ret > largest):
      largest = largest_ret
      largest_inner = inner
      largest_outer = outer

print(largest_outer, largest_inner, largest)
