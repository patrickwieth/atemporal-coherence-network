import random
import numpy as np
import pathos.multiprocessing as mp 

p = mp.ProcessingPool(2)
search_iterations = 10

def read_parameters():
	return np.load('results/parameter.npy')

def dump_parameters(paras):
	np.save("results/parameter.npy", paras)



def evaluate_fitness(evaluation_function, parameter):
	samples = range(200)
	fitness = 0

	testf = lambda x: evaluation_function(parameter)

	fitness = p.map(testf, samples)
    

	return np.sum(fitness) / len(samples)

def optimize_single_parameter(evaluation_function, parameter_list, index_to_optimize):
	i = index_to_optimize
	old_optimum = parameter_list[i, 0]
	max_fitness = [0, -1]
	for j in range(search_iterations):
		# set single parameter according to current iteration
		parameter_list[i, 0] = parameter_list[i, 1] + j*(parameter_list[i, 2] - parameter_list[i, 1])/search_iterations
		
		fitness = evaluate_fitness(evaluation_function, parameter_list[:,0])

		if(fitness > max_fitness[1]): max_fitness = [parameter_list[i, 0], fitness]

	# better parameter found? update...
	if(old_optimum != max_fitness[0]): 
		parameter_list[i, 0] = max_fitness[0]
	# old parameter remains the best then search less broad but finer
	else: 
		parameter_list[i, 0] = old_optimum

		# is the parameter in the lower third of the boundaries?
		if(parameter_list[i, 0] - parameter_list[i, 1] < (parameter_list[i, 2] - parameter_list[i, 1]) / 3):
			# then decrease the upper boundary
			parameter_list[i, 2] -=  0.5 * (parameter_list[i, 2] - parameter_list[i, 1])
		# is the parameter in the upper third of the boundaries?
		elif(parameter_list[i, 0] - parameter_list[i, 1] > 2 * (parameter_list[i, 2] - parameter_list[i, 1]) / 3):
			# then increase the lower boundary
			parameter_list[i, 1] +=  0.5 * (parameter_list[i, 2] - parameter_list[i, 1])
		# is it in the middle?
		else:
			# then decrease the upper boundary and incrase the lower boundary
			parameter_list[i, 2] -=  0.25 * (parameter_list[i, 2] - parameter_list[i, 1])
			parameter_list[i, 1] +=  0.25 * (parameter_list[i, 2] - parameter_list[i, 1])

	print("best:", max_fitness)

	return parameter_list


def sweep(evaluation_function, parameter_list):
	for i, param in enumerate(parameter_list[:,0]):
		parameter_list = optimize_single_parameter(evaluation_function, parameter_list, i)

	return parameter_list


def run(evaluation_function, parameter_list):

	base_parameter_list = read_parameters()
	print(base_parameter_list)
	parameter_list = np.copy(base_parameter_list)
	
	while True:
		parameter_list = sweep(evaluation_function, parameter_list)
		print(parameter_list)
		dump_parameters(parameter_list)

