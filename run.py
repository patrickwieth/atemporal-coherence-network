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
rolled_mechanisms = creator.roll_dice()
print("behavior:", rolled_mechanisms)

#behavior = [creator.create_scheme(rolled_mechanisms)]
behavior = [schemes.base_scheme]	# for base scheme usage, uncomment this


def evalOne(individual):
	parameters = behavior.individual_to_parameter(individual)

	topology = network.architecture.topology(number_of_neurons)	
	net = network.architecture.instance(topology, behavior, parameters)
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
print(behavior.parameters)

test1 = evolution.search.evolution(evalOne, behavior.get_parameter_dict())

seconds_to_go = 10
running_for = 0
gen = 1
start = time.time()

while running_for < seconds_to_go:
	print("running generation #"+str(gen))
	test1.iterate()

	gen += 1
	running_for = time.time() - start

new_best_species = test1.get_best(5)
print("performance of top 5", new_best_species[:,1])
print("after", running_for, "seconds")



print("applying for hall of fame")
hall_of_basic_pattern = evolution.hall_of_fame.load("basic_pattern")

heroes = hall_of_basic_pattern.individuals
heroes_performance = []


for x in heroes:
	loaded_scheme = schemes.scheme()
	
	mechanism_list = [network.mechanisms.registered[y] for y in x.mechanisms]

	#print(mechanism_list)

	loaded_scheme.set_mechanisms_by_list(mechanism_list)

	print(x.mechanisms, x.parameters, x.score)
	print(loaded_scheme.individual_to_parameter(x.parameters))
	heroes_performance.append(evalOne(loaded_scheme.individual_to_parameter(x.parameters)))

print(creator.mechanisms_to_genes(rolled_mechanisms))

if len(heroes_performance) == 0:
	new_hero = evolution.individuum.individuum(creator.mechanisms_to_genes(rolled_mechanisms), new_best_species[-1,0], new_best_species[-1,1])
	hall_of_basic_pattern.insert_individual(new_hero)	

print(hall_of_basic_pattern.individuals[0].score)

hall_of_basic_pattern.save()

#np.sort(heroes_performance)

'''
'''