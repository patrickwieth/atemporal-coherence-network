import random
#import matplotlib.pyplot as plt
import numpy as np
import pathos.multiprocessing as mp 
import network
import grid_search

number_of_neurons = 3

# default parameters describe: threshold, [lower and upper init weight], [weight_boost, weight_penalty] and [activation_boost, activation_penalty, activation_diminishing]
default_parameter = network.architecture.parameter(0.2, [0.5, 2], [0.02, 0.02], [0, 0.02, 0.99], 0.99)
# default topology describes: size
default_topology = network.architecture.topology(number_of_neurons)



def prepare_data():

	input_a = [0, 1, 0, 1, 0, 1, 0, 1]
	input_b = [1, 0, 1, 0, 1, 0, 1, 0]
	input_c = [1, 1, 0, 0, 1, 1, 0, 0]
	input_d = [0, 0, 1, 1, 0, 0, 1, 1]

	def rand_input():
		return [random.uniform(0, 2) for _ in range(8)]
		#return [random.uniform(0,2), random.uniform(0,2), random.uniform(0,2), random.uniform(0,2), random.uniform(0,2), random.uniform(0,2), random.uniform(0,2), random.uniform(0,2)]

	def add_noise(data):
		f = lambda x : x + random.uniform(-x/5, x/5)
		return(list(map(f, data)))

	data = []
	data.append(input_a)
	data.append(input_b)
	data.append(input_c)

	for i in range(1000):
		pick = random.randint(0, number_of_neurons)
		print(pick)
		if(pick == 0):	
			data.append(rand_input())
			#data.append(rand_input())
		elif(pick == 1):
			data.append(add_noise(input_a))
		elif(pick == 2):
			data.append(add_noise(input_b))
		else:
			data.append(add_noise(input_c))
	return np.array(data)



# start, lower search, upper search values
startparas = np.array([
	[0.3,	0,	1],			#threshold
	[1.6,	0,	2],			#lower_init_weight
	[0.8,	0,	5],			#upper_init_weight
	[0.9,	0,	1],			#weight_boost
	[0.3,	0,	1],			#weight_penalty
	[0.3,	0,	1],			#activation_boost
	[0.01,	0,0.1],			#activation_penalty
	[0.3,	0,	1],			#activation_diminishing
	[0.7,	0,  1]])		#intercon_diminishing




def set_eval_func():
	data = prepare_data()
	topology = default_topology
	fitness_f = network.fitness.discrimination
	
	n = number_of_neurons

	def eval_func(parameter):
		parameter = network.architecture.flat_array_to_parameter(parameter)
		net = network.architecture.instance(topology, parameter)

		samples = 200
		net.run(data, samples)
		result = net.test(data, n)

		return fitness_f(result)

	return eval_func


#grid_search.dump_parameters(startparas)

grid_search.run(set_eval_func(), [])


def testrun(parameter):
	data = prepare_data()
	topology = default_topology
	parameter = network.architecture.flat_array_to_parameter(parameter)
	net = network.architecture.instance(topology, parameter)

	samples = 200
	net.run(data, samples)
	result = net.test(data, number_of_neurons)
	fitness = network.fitness.discrimination(result)

	print(result)
	print(fitness)

	return 0


paras = grid_search.read_parameters()
print(paras)
testrun(paras[:,0])


plotdata = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]


#plt.scatter(plotdata[0], plotdata[1], plotdata[2])
#plt.show()
