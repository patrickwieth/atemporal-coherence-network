import os.path
import numpy as np
from evolution import individuum

#########################################################################################################
# The hall of fame is a class, which keeps track of the best species 									#
# A number of famous species can be defined, for which the genotype and parameter config is stored.		#
# Also this can be saved to hdd and be reloaded, so that from these beings new searches can be started 	#
#########################################################################################################

class hall_of_fame:
	def __init__(self, name, description, number_of_individuals):
		self.name = name
		self.description = description
		self.number_of_individuals = number_of_individuals
		self.individuals = []

	def insert_individual(self, individuum):
		superior = False

		for x in self.individuals:
			if superior:
				temp = x
				x = last
				last = temp
			elif individuum.score > x.score:
				superior = True
				last = x
				x = individuum

		print("individuals:", len(self.individuals), "max:", self.number_of_individuals)
		if not superior and len(self.individuals) < int(self.number_of_individuals):
			self.individuals.append(individuum)

	def save(self):
		genes = np.array([[x.mechanisms, x.parameters, x.score] for x in self.individuals])
		data = np.array([self.name, self.description, self.number_of_individuals, genes])
		np.save('results/hall_of_'+self.name+'.npy', data)

def load(name):
	

	if os.path.isfile('results/hall_of_'+name+'.npy') == False:
		number_of_individuals = 5
		new_hall = hall_of_fame(name, "fresh hall of fame", number_of_individuals)
		new_hall.save()
		return new_hall

	data = np.load('results/hall_of_'+name+'.npy')
	loaded = hall_of_fame(data[0], data[1], data[2])

	if isinstance(data[3], str):
		return loaded

	for x in data[3]:
		new_in = individuum.individuum(x[0], x[1], x[2])
		loaded.insert_individual(new_in)

	return loaded
