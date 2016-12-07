import network.fitness as fitness
import numpy as np

a = np.array(
	[[1,2,3],
	 [4,5,6],	
	 [7,8,9]])

b = np.array(
	[[1,0, 0],
	[0,	1, 0],	
	[0,	0, 1]])

c = np.array(
	[[1,1, 1],
	[1,	1, 0],	
	[1,	0, 1]])

d = np.array(
	[[1,1, 1],
	[0,	1, 0],	
	[0,	0, 1]])

print(fitness.discrimination(a)) # should be below 0
print(fitness.discrimination(b)) # should be 1
print(fitness.discrimination(c)) # should be below 0.5
print(fitness.discrimination(d)) # should be around 0.5