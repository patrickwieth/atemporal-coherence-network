import evolution
import network
import util

number_of_neurons = 4

data = util.data.patterns(number_of_neurons, 0.1, 0.1)

def evalOneMax(individual):
	params = network.architecture.flat_array_to_parameter(individual)
	topology = network.architecture.topology(number_of_neurons)
	net = network.architecture.instance(topology, params)
	net.run(data, 200)
	result = net.test(data, number_of_neurons)

	return network.fitness.discrimination(result),



test1 = evolution.search.evolution(evalOneMax)


NGEN=1
for gen in range(NGEN):
	print("running generation #"+str(gen))
	test1.iterate()

test1.get_best()


#evolution.search.single_run()