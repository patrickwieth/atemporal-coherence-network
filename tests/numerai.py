import random
import numpy as np
import network

import os
cwd = os.getcwd()

print(cwd)

data = np.genfromtxt('../numerai/data/46/numerai_training_data.csv', delimiter=',')

training_to = 90000
training_data = data[1:, :-1] #data[1:training_to]


def input_prepration():
	raw_input = training_data[random.randint(0, len(training_data)-1)]		

	inv_f = lambda x: 1/x if x != 0 else 0
	inv_input = np.array(list(map(inv_f, raw_input)))

	return np.array([raw_input, inv_input]).flatten()
	

def testrun():
	#data = prepare_data()

	default_topology = network.architecture.topology(2)
	default_parameter = network.architecture.parameter(0.3272, [0.150, 4.115], [0.7165, 0.0477], [0.3213, 0.0177, 0.2224], 0.86)
	
	net = network.architecture.instance(default_topology, default_parameter)

	samples = 100000
	net.run(training_data, samples)
	result = net.test(training_data, 10)

	result.append(data[1:11,-1])

	
	#fitness = discrimination_fitness(norm(result))

	print(np.transpose(np.array(result)))
	#print(fitness)

	return 0

testrun()