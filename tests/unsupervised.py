import random
import matplotlib.pyplot as plt
import numpy as np
import network

# default parameters describe: threshold, [lower and upper init weight], [weight_boost, weight_penalty] and [activation_boost, activation_penalty, activation_diminishing]
default_parameter = network.architecture.parameter(0.2, [0.5, 2], [0.02, 0.02], [0, 0.02, 0.99], 0.99)
# default topology describes: size
default_topology = network.architecture.topology(2)


def discrimination_fitness(result):
	return 1/4*(abs(result[0][0] - result[0][1]) + abs(result[1][0] - result[1][1]) + abs(result[0][0] - result[1][0]) + abs(result[0][1] - result[1][1]))

def norm(data):
	maxval = max(abs(data[0][0]), abs(data[1][0]), abs(data[0][1]), abs(data[1][1]))
	if(maxval == 0): return data 	
	f = lambda x : x/abs(maxval)
	g = lambda x : list(map(f, x))

	return(list(map(g, data)))

def read_parameters():
	return np.load('tests/parameter.npy')

def dump_parameters(paras):
	np.save("tests/parameter.npy", paras)

def prepare_data():

	input_a = [0, 1, 0, 1]
	input_b = [1, 0, 1, 0]
	input_c = [1, 1, 0, 0]

	def rand_input():
		return [random.uniform(0,2), random.uniform(0,2), random.uniform(0,2), random.uniform(0,2)]

	def add_noise(data):
		f = lambda x : x + random.uniform(-x/5, x/5)
		return(list(map(f, data)))

	data = []
	data.append(input_a)
	data.append(input_b)

	for i in range(1000):
		pick = random.randint(0,2)
		if(pick == 0):
			data.append(add_noise(input_a))
		elif(pick == 1):
			data.append(add_noise(input_b))
		else:
			data.append(rand_input())
			data.append(rand_input())

	return data

def evaluate(data, i, j):
	samples = 300
	fitness = 0

	for k in range(samples):
											# threshold, [lower and upper init weight], [weight_boost, weight_penalty] and [activation_boost, activation_penalty, activation_diminishing], intercon_diminishing
		parameter = network.architecture.parameter(0.713,	 [0.276, 0.28 + j/1000],				[0.38, 			0.049],				[0.017, 				0.04 + i/1000, 				0.918], 				0.833)

		net = network.architecture.instance(default_topology, parameter)
		result = net.run(data, min(10+samples, 200))
		
		fitness += discrimination_fitness(norm(result))

	print(fitness / samples)
	return fitness / samples


def test():
	
	data = prepare_data()

	results = []
	maxfit = [0, -1, -1]
	for i in range(11):
		print("i:", i)
		for j in range(10):
			fitness = evaluate(data, i-5, j-5)
			if(fitness > maxfit[0]):
				maxfit = [fitness, i-5, j-5]
			results.append([i, j, 300*fitness**4])	

	print("best result: ", maxfit)

	plotdata = np.transpose(np.array(results))

	plt.scatter(plotdata[0], plotdata[1], plotdata[2])
	plt.show()


#test()




def grid_search():

	search_iterations = 10

	def evaluate_fitness(net, data):
		samples = 400
		fitness = 0
		for k in range(samples):								
			result = net.run(data, min(10+samples, 200))
			fitness += discrimination_fitness(norm(result))
		return fitness / samples

	def optimize_single_parameter(parameter_list, index_to_optimize):
		i = index_to_optimize
		old_optimum = parameter_list[i, 0]
		max_fitness = [0, -1]
		for j in range(search_iterations):
			# set single parameter according to current iteration
			parameter_list[i, 0] = parameter_list[i, 1] + j*(parameter_list[i, 2] - parameter_list[i, 1])/search_iterations

			#print("creating with parameter:", parameter_list[:,0])
			# spawn network
			parameter = network.architecture.flat_array_to_parameter(parameter_list[:,0])
			net = network.architecture.instance(default_topology, parameter)

			# evaluate network
			fitti = evaluate_fitness(net, data)
			if(fitti > max_fitness[1]): max_fitness = [parameter_list[i, 0], fitti]

			#print(parameter_list[i, 0], ": ", fitti)

		# better parameter found? update...
		if(max_fitness[1] > -1): 
			parameter_list[i, 0] = max_fitness[0]
		# old parameter remains the best then search less broad but finer
		else: 
			parameter_list[i, 0] = old_optimum

			print("old optimum")

			# is the parameter in the lower third of the boundaries?
			if(parameter_list[i, 0] - parameter_list[i, 1] < (parameter_list[i, 2] - parameter_list[i, 1]) / 3):
				print("decrease upper")
				# then decrease the upper boundary
				parameter_list[i, 2] -=  0.5 * (parameter_list[i, 2] - parameter_list[i, 1])
			# is the parameter in the upper third of the boundaries?
			elif(parameter_list[i, 0] - parameter_list[i, 1] > 2 * (parameter_list[i, 2] - parameter_list[i, 1]) / 3):
				print("increase lower")
				# then increase the lower boundary
				parameter_list[i, 1] +=  0.5 * (parameter_list[i, 2] - parameter_list[i, 1])
			# is it in the middle?
			else:
				print("change both")
				# then decrease the upper boundary and incrase the lower boundary
				parameter_list[i, 2] -=  0.25 * (parameter_list[i, 2] - parameter_list[i, 1])
				parameter_list[i, 1] +=  0.25 * (parameter_list[i, 2] - parameter_list[i, 1])

		print("best:", max_fitness)

		return parameter_list

	data = prepare_data()

	

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

	base_parameter_list = read_parameters()

	print(base_parameter_list)

	parameter_list = np.copy(base_parameter_list)

	def sweep(parameter_list):

		for i, param in enumerate(parameter_list[:,0]):
			print("param #", i)
			parameter_list = optimize_single_parameter(parameter_list, i)

		return parameter_list

	while True:
		parameter_list = sweep(parameter_list)
		print(parameter_list)
		dump_parameters(parameter_list)


grid_search()