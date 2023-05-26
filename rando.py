import math
import random

random.seed(20)

t = random.sample(range(-10, 10), 10)
t = t*random.random()
print(t)

p = []
for i in range(0,1000):
    p.append(random.random())

