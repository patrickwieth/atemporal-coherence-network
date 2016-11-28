import random
#import matplotlib.pyplot as plt
import numpy as np
import pathos.multiprocessing as mp 
import network
import grid_search



# default parameters describe: threshold, [lower and upper init weight], [weight_boost, weight_penalty] and [activation_boost, activation_penalty, activation_diminishing]
default_parameter = network.architecture.parameter(0.2, [0.5, 2], [0.02, 0.02], [0, 0.02, 0.99], 0.99)
# default topology describes: size
default_topology = network.architecture.topology(4)


def norm(data):
	maxval = np.amax(data)
	if(maxval == 0):	return data 	
	else:				return data / maxval

def discrimination_fitness(matrix):

	def discriminate(line):
		maxi = np.argmax(line)

		sum = 0
		for i, val in enumerate(line):
			if(i == maxi): 	sum += val
			else: 			sum -= val

		return sum

	row_sums = list(map(discriminate, matrix))
	col_sums = list(map(discriminate, np.transpose(matrix)))

	return (np.sum(row_sums) + np.sum(col_sums)) / (len(row_sums) + len(col_sums))


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
	data.append(input_d)

	for i in range(1000):
		pick = random.randint(0,4)
		if(pick == 0):
			data.append(add_noise(input_a))
		elif(pick == 1):
			data.append(add_noise(input_b))
		elif(pick == 2):
			data.append(add_noise(input_c))
		elif(pick == 3):
			data.append(add_noise(input_d))
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





def set_eval_func():
	data = prepare_data()
	topology = default_topology
	fitness_f = discrimination_fitness
	norm_f = norm

	def eval_func(parameter):
		parameter = network.architecture.flat_array_to_parameter(parameter)
		net = network.architecture.instance(topology, parameter)

		samples = 200
		result = net.run(data, samples)
		return fitness_f(norm_f(result))

	return eval_func


#grid_search.dump_parameters(startparas)

#grid_search.run(set_eval_func(), [])


def testrun(parameter):
	data = prepare_data()
	topology = default_topology
	parameter = network.architecture.flat_array_to_parameter(parameter)
	net = network.architecture.instance(topology, parameter)

	samples = 10000
	result = net.run(data, samples)
	#fitness = discrimination_fitness(norm(result))

	print(np.array(result))
	#print(fitness)

	return 0


paras = grid_search.read_parameters()
print(paras)
testrun(paras[:,0])