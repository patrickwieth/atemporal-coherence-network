import random
import numpy as np
from deap import creator, base, tools, algorithms
import network
import util

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=9)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

number_of_neurons = 4

data = util.data.patterns(number_of_neurons, 0.1, 0.1)

def evalOneMax(individual):
	params = network.architecture.flat_array_to_parameter(individual)
	topology = network.architecture.topology(number_of_neurons)
	net = network.architecture.instance(topology, params)
	net.run(data, 200)
	result = net.test(data, number_of_neurons)

	return network.fitness.discrimination(result),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=300)

NGEN=100
for gen in range(NGEN):
	offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
	fits = toolbox.map(toolbox.evaluate, offspring)
	for fit, ind in zip(fits, offspring):
		#print(fit)
		ind.fitness.values = fit
	population = toolbox.select(offspring, k=len(population))


top10 = tools.selBest(population, k=10)

#print(top10)

fitnesses = list(map(evalOneMax,top10))

for idx, val in enumerate(top10):
	print("params:", val)
	print("fitness", fitnesses[idx], "\n")

