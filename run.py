import evolution
import network
import util
import time

from network import schemes

number_of_neurons = 4

data = util.data.patterns(number_of_neurons, 0.01, 0.02)

def evalOneMax(individual):
	params = network.architecture.flat_array_to_parameter(individual)
	topology = network.architecture.topology(number_of_neurons)
	net = network.architecture.instance(topology, [schemes.base_scheme], params)
	net.run(data, 200)
	result = net.test(data, number_of_neurons)

	return network.fitness.discrimination(result),


test1 = evolution.search.evolution(evalOneMax)

seconds_to_go = 5
running_for = 0
gen = 1
start = time.time()


while running_for < seconds_to_go:
	print("running generation #"+str(gen))
	test1.iterate()

	gen += 1
	running_for = time.time()-start

res = test1.get_best(10)

print(res[:,1])

print("after ", running_for, "seconds")

