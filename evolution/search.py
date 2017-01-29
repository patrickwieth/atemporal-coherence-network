import random
import numpy as np
from deap import creator, base, tools, algorithms
import network
import util

class evolution:
	def __init__(self, evaluation_function, parameter_frame):

		creator.create("FitnessMax", base.Fitness, weights=(1.0,))
		creator.create("Individual", list, fitness=creator.FitnessMax)

		self.toolbox = base.Toolbox()

		# parameters are defined here
		attributes = []

		for key, value in parameter_frame.items():
			self.toolbox.register(key, random.uniform, value[0], value[1])
			attributes += [getattr(self.toolbox, key)]

		self.toolbox.register("individual", tools.initCycle, creator.Individual, tuple(attributes), n=len(parameter_frame))
		self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

		self.toolbox.register("evaluate", evaluation_function)
		self.toolbox.register("mate", tools.cxTwoPoint)
		self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
		self.toolbox.register("select", tools.selTournament, tournsize=3)

		self.population = self.toolbox.population(n=20)

	def iterate(self):
		offspring = algorithms.varAnd(self.population, self.toolbox, cxpb=0.5, mutpb=0.1)
		fits = self.toolbox.map(self.toolbox.evaluate, offspring)
		for fit, ind in zip(fits, offspring):
			ind.fitness.values = fit
				
		self.population = self.toolbox.select(offspring, k=len(self.population))

	def get_best(self, count):
		top = tools.selBest(self.population, k=count)
		fitnesses = list(map(self.toolbox.evaluate, top))
		result = np.array(list(zip(top, [f[0] for f in fitnesses])))

		return np.sort(result, axis=0)
