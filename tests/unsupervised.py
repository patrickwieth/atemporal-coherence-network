import random

import network

# default parameters describe: threshold, [lower and upper init weight], [weight_boost, weight_penalty, weight_diminishing] and [activation_boost, activation_penalty, activation_diminishing]
default_parameter = network.architecture.parameter(0.2, [0.5, 2], [0.02, 0.02, 0.99], [0, 0.02, 0.99])

default_topology = network.architecture.topology(2)

def klar():

	
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

	for i in range(100):
		ne_is_ok = network.architecture.instance(default_topology, default_parameter)
		result = ne_is_ok.run(data, 100)

		maxval = max(result[0][0], result[1][0], result[0][1], result[1][1])
		#print(maxval)

		for i in result:
			for j in i:
				print(j)
				j = j/maxval

		result_fitness = abs(result[0][0] * result[1][1] - result[0][1] * result[1][0]) - abs(result[0][0] * result[0][1]) - abs(result[1][0] * result[1][1])

		results.append(result_fitness)

	print(results)
	

	print("{:f} {:f} \n{:f} {:f}".format(result[0][0], result[0][1], result[1][0], result[1][1]))

	

klar()


