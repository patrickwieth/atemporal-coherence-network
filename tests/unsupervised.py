import random
import matplotlib.pyplot as plt
import numpy as np
import network

# default parameters describe: threshold, [lower and upper init weight], [weight_boost, weight_penalty, weight_diminishing] and [activation_boost, activation_penalty, activation_diminishing]
default_parameter = network.architecture.parameter(0.2, [0.5, 2], [0.02, 0.02, 0.99], [0, 0.02, 0.99], 0.99)
# default topology describes: size
default_topology = network.architecture.topology(2)


def discrimination_fitness(result):
	return 1/4*(abs(result[0][0] - result[0][1]) + abs(result[1][0] - result[1][1]) + abs(result[0][0] - result[1][0]) + abs(result[0][1] - result[1][1]))

def norm(data):
	maxval = max(data[0][0], data[1][0], data[0][1], data[1][1])
	if(maxval == 0): return data 	
	#for l in data:
	#	for m in l:
	#		m = m/maxval
	#		print(m)
	data[0][0] /= maxval
	data[1][0] /= maxval
	data[0][1] /= maxval
	data[1][1] /= maxval
	return data

def evaluate(data, i, j):
	samples = 100
	fitness = 0
	for k in range(samples):
		# threshold, [lower and upper init weight], [weight_boost, weight_penalty, weight_diminishing] and [activation_boost, activation_penalty, activation_diminishing]
		parameter = network.architecture.parameter(0.2 + j/10, [0.5, 2], [0.01, 0.01 + i/100, 0.99], [0, 0.02, 0.99], 0.99)

		net = network.architecture.instance(default_topology, parameter)
		result = net.run(data, 100)
		
		fitness += discrimination_fitness(norm(result))

	print(fitness / samples)
	return fitness / samples


def test():
	
	input_a = [0, 1, 0, 1]
	input_b = [1, 0, 1, 0]

	def rand_input():
		return [random.randrange(0,1), random.randrange(0,1), random.randrange(0,1), random.randrange(0,1)]

	data = []

	data.append(input_a)
	data.append(input_b)

	for i in range(100):
		pick = random.randint(0,2)
		if(pick == 0):
			data.append(input_a)
		elif(pick == 1):
			data.append(input_b)
		else:
			data.append(rand_input())

	results = []
	for i in range(20):
		print("i:", i)
		for j in range(20):
			fitness = evaluate(data, i, j)
			results.append([i, j, (30*fitness)**2])	

	plotdata = np.transpose(np.array(results))

	plt.scatter(plotdata[0], plotdata[1], plotdata[2])
	plt.show()


test()
