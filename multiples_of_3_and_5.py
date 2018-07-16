#!/usr/bin/python

list_of_multiples = []
sum_of_multiples = 0

for outer in range(3, 1000):
  # Test for a multple of 3 or 5
  if outer%3 == 0:
    sum_of_multiples = sum_of_multiples + outer
  elif outer%5 == 0:
    sum_of_multiples = sum_of_multiples + outer

# Print our results
print("sum of multiples of 3 or 5 below 1000 = " + str(sum_of_multiples))
