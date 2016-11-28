import random
#import matplotlib.pyplot as plt
import numpy as np
import pathos.multiprocessing as mp 
import network
import grid_search



# default parameters describe: threshold, [lower and upper init weight], [weight_boost, weight_penalty] and [activation_boost, activation_penalty, activation_diminishing]
default_parameter = network.architecture.parameter(0.2, [0.5, 2], [0.02, 0.02], [0, 0.02, 0.99], 0.99)
# default topology describes: size
default_topology = network.architecture.topology(3)


def discrimination_fitness(result):
	return 1/4*(abs(result[0][0] - result[0][1]) + abs(result[1][0] - result[1][1]) + abs(result[0][0] - result[1][0]) + abs(result[0][1] - result[1][1]))

def norm(data):
	maxval = max(abs(data[0][0]), abs(data[1][0]), abs(data[0][1]), abs(data[1][1]))
	if(maxval == 0): return data 	
	f = lambda x : x/abs(maxval)
	g = lambda x : list(map(f, x))

	return(list(map(g, data)))

def prepare_data():

	input_a = [0, 1, 0, 1]
	input_b = [1, 0, 1, 0]
	input_c = [1, 1, 0, 0]
	input_d = [0, 0, 1, 1]

	def rand_input():
		return [random.uniform(0,2), random.uniform(0,2), random.uniform(0,2), random.uniform(0,2)]

	def add_noise(data):
		f = lambda x : x + random.uniform(-x/5, x/5)
		return(list(map(f, data)))

	data = []
	data.append(input_a)
	data.append(input_b)
	data.append(input_c)

	for i in range(1000):
		pick = random.randint(0,3)
		if(pick == 0):
			data.append(add_noise(input_a))
		elif(pick == 1):
			data.append(add_noise(input_b))
		elif(pick == 2):
			data.append(add_noise(input_c))
		else:
			data.append(rand_input())
			data.append(rand_input())

	return data


'''
											# threshold, [lower and upper init weight], [weight_boost, weight_penalty] and [activation_boost, activation_penalty, activation_diminishing], intercon_diminishing
		parameter = network.architecture.parameter(0.713,	 [0.276, 0.28 + j/1000],				[0.38, 			0.049],				[0.017, 				0.04 + i/1000, 				0.918], 				0.833)
'''


#plt.scatter(plotdata[0], plotdata[1], plotdata[2])
#plt.show()


# start, lower search, upper search values
'''
	threshold 				= [0.30,	0,	0.5]
	lower_init_weight 		= [0.00,	0,	1]
	upper_init_weight	 	= [2.75,	1.25,	3.75]
	weight_boost 			= [0.20,	0,	1]
	weight_penalty 			= [0.10,	0,	0.125]
	activation_boost	 	= [0.10,	0,	1]
	activation_penalty		= [0.35,	0,	0.5]
	activation_diminishing	= [0.05,	0,	0.5]
	intercon_diminishing	= [0.70,	0.5, 1]
'''

startparas = np.array([
	[0.30,	0,	1],
	[0.00,	0,	1],
	[2.75,	0,	5],
	[0.20,	0,	1],
	[0.10,	0,	1],
	[0.10,	0,	1],
	[0.35,	0,	1],
	[0.05,	0,	1],
	[0.70,	0,  1]])




def set_eval_func():
	data = prepare_data()
	topology = default_topology
	fitness_f = discrimination_fitness
	norm_f = norm

	def eval_func(parameter):
		parameter = network.architecture.flat_array_to_parameter(parameter)
		net = network.architecture.instance(topology, parameter)

		samples = 200
		result = net.run(data, min(10+samples, 200))
		return fitness_f(norm_f(result))

	return eval_func


grid_search.dump_parameters(startparas)

grid_search.run(set_eval_func(), [])


