import random
#import matplotlib.pyplot as plt
import numpy as np
import pathos.multiprocessing as mp 
import network
import grid
import util

number_of_neurons = 2

# default parameters describe: threshold, [lower and upper init weight], [weight_boost, weight_penalty] and [activation_boost, activation_penalty, activation_diminishing]
default_parameter = network.architecture.parameter(0.2, [0.5, 2], [0.02, 0.02], [0, 0.02, 0.99], 0.99)
# default topology describes: size
default_topology = network.architecture.topology(number_of_neurons)

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
	n = number_of_neurons

	data = util.data.patterns(n, 0.01, 0.2)

	#print(data)
	#for i in data:
	#	print(i)

	topology = default_topology
	fitness_f = network.fitness.discrimination

	def eval_func(parameter):
		parameter = network.architecture.flat_array_to_parameter(parameter)
		net = network.architecture.instance(topology, parameter)

		samples = 200
		net.run(data, samples)
		result = net.test(data, n)

		return fitness_f(result)

	return eval_func


#util.io.dump_parameters(startparas)

grid.search.run(set_eval_func(), [])




def testrun(parameter):
	n = number_of_neurons

	data = util.data.patterns(n, 0.0, 0.2)
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
