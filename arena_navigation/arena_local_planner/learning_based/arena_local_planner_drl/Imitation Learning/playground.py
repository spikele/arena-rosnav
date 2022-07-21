"""products = {'t121': {'name': '42" Sony TV', 'brand': 'Sony', 'price':600},
            'c702': {'name': 'Camera 8989', 'brand': 'Cannon', 'price':400}}

# Add new dictionary
products['f326'] = {'name': 'Fridge', 'brand': 'LG', 'price': 700}

# Print the keys and values of the dictionary after insertion
for pro in products:
  print('Name:',products[pro]['name'],', '
        'Brand:',products[pro]['brand'], ', '
        'Price:$',products[pro]['price'])

for key in products:
    print("products key: " + str(key))
print(products.keys())"""
from re import A
import torch
import numpy as np


a = np.array([1, 2, 3, 4, 5, 6, 7])
b = np.array([1, 2, 3, 4, 5, 6, 7])


def multiply(a, b):
  return a*b

c = multiply(a[:],b[:])

print(a)
print(b)
print(c)