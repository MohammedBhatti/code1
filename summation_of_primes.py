#!/usr/bin/python

list_of_primes = []
sum_of_primes = 0

for outer in range(2, 2000):

  # Assume outer number is prime
  prime = True
  for inner in range(2, outer):
    # Not a prime
    if outer % inner == 0:
      prime = False
      # Don't test anymore, just break
      break
    # Break when inner is greater than 1/2 outer
    if inner >= outer//2:
      break

  # Sum our primes and add them to our list for checking
  if prime:
    list_of_primes.append(outer)
    sum_of_primes = sum_of_primes + outer

for i in list_of_primes:
  print(i)

# Print our results
print("sum of primes = " + str(sum_of_primes))
