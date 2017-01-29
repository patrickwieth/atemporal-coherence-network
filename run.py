import evolution
import network
import util
import time

from network import schemes
from network import scheme_crafting

number_of_neurons = 4

data = util.data.patterns(number_of_neurons, 0.1, 0.2)
print("first data points:\n", data[0:number_of_neurons])

creator = scheme_crafting.scheme_dice()
rollit = creator.roll_dice()
print("behavior:", rollit)

behavior = [creator.create_scheme(rollit)]

#behavior = [schemes.base_scheme]	# for base scheme usage, uncomment this


def evalOneMax(individual):
	params = network.architecture.flat_array_to_parameter(individual)

	topology = network.architecture.topology(number_of_neurons)
	
	net = network.architecture.instance(topology, behavior, params)
	net.run(data, 200)
	result = net.test(data, number_of_neurons)

	return network.fitness.discrimination(result),

def merge_schemes(schemes):
	if(len(schemes) > 1):
		return schemes[0].integrate_schemes(schemes[1:])
	else:
		return schemes[0]		

behavior = merge_schemes(behavior)

print("number of parameters:", len(behavior.parameters))


test1 = evolution.search.evolution(evalOneMax, behavior.parameters)

seconds_to_go = 5
running_for = 0
gen = 1
start = time.time()

while running_for < seconds_to_go:
	print("running generation #"+str(gen))
	test1.iterate()

	gen += 1
	running_for = time.time() - start

res = test1.get_best(10)
print("performance of top 10", res[:,1])
print("after", running_for, "seconds")

