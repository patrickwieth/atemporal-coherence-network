import random
import matplotlib.pyplot as plt
import numpy as np

import network

# default parameters describe: threshold, [lower and upper init weight], [weight_boost, weight_penalty, weight_diminishing] and [activation_boost, activation_penalty, activation_diminishing]
default_parameter = network.architecture.parameter(0.2, [0.5, 2], [0.02, 0.02, 0.99], [0, 0.02, 0.99])
# default topology describes: size
default_topology = network.architecture.topology(2)



def evaluate(data, i, j):
	samples = 100
	fitness = 0
	for i in range(samples):
		parameter = network.architecture.parameter(0.2, [0.5, 2], [0.00 + i/10, 0.00 + j/10, 0.99], [0, 0.02, 0.99])

		net = network.architecture.instance(default_topology, default_parameter)
		result = net.run(data, 100)

		maxval = max(result[0][0], result[1][0], result[0][1], result[1][1])
		for i in result:
			for j in i:
				j = j/maxval

		fitness += abs(result[0][0] * result[1][1] - result[0][1] * result[1][0]) - abs(result[0][0] * result[0][1]) - abs(result[1][0] * result[1][1])
		print(fitness)

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
	for i in range(4):
		for j in range(5):
			fitness = evaluate(data, i, j)

			results.append([i, j, 1*fitness**3])
	

	plotdata = np.transpose(np.array(results))


	plt.scatter(plotdata[0], plotdata[1], plotdata[2])
	plt.show()


test()
