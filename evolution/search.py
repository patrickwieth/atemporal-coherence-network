import random
import numpy as np
from deap import creator, base, tools, algorithms
import network
import util

#number_of_neurons = 4

#data = util.data.patterns(number_of_neurons, 0.1, 0.1)

class evolution:
	def __init__(self, evaluation_function):
		creator.create("FitnessMax", base.Fitness, weights=(1.0,))
		creator.create("Individual", list, fitness=creator.FitnessMax)

		self.toolbox = base.Toolbox()

		self.toolbox.register("attr_float", random.uniform, 0, 1)
		self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=9)
		self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

		self.toolbox.register("evaluate", evaluation_function)
		self.toolbox.register("mate", tools.cxTwoPoint)
		self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
		self.toolbox.register("select", tools.selTournament, tournsize=3)

		self.population = self.toolbox.population(n=300)

	def iterate(self):
		offspring = algorithms.varAnd(self.population, self.toolbox, cxpb=0.5, mutpb=0.1)
		fits = self.toolbox.map(self.toolbox.evaluate, offspring)
		for fit, ind in zip(fits, offspring):
			ind.fitness.values = fit
				
		self.population = self.toolbox.select(offspring, k=len(self.population))

	def get_best(self):
		top10 = tools.selBest(self.population, k=10)

		fitnesses = list(map(self.toolbox.evaluate, top10))
		
		print(fitnesses)

		for idx, val in enumerate(top10):
			print("params:", val)
			print("fitness", fitnesses[idx], "\n")

