import random
from network import neurons

class parameter:
	def __init__(self, threshold, init_weight_bounds, weight_ajdustors, activation_adjustors):
		self.threshold = threshold
		self.init_lower_weight = init_weight_bounds[0]
		self.init_upper_weight = init_weight_bounds[1]
		
		self.weight_boost = weight_ajdustors[0]
		self.weight_penalty = weight_ajdustors[1]
		self.weight_diminishing = weight_ajdustors[2]
		self.activation_boost = activation_adjustors[0]
		self.activation_penalty = activation_adjustors[1]
		self.activation_diminishing = activation_adjustors[2]


class topology:
	def __init__(self, size):
		self.size = size

class instance:
	def __init__(self, topology, parameter):

		self.neurons = []

		for i in range(topology.size):
			self.neurons.append(neurons.connectron(parameter))

		for i, a in enumerate(self.neurons):
			for j, b in enumerate(self.neurons):
				if(i != j):
					a.set_interconnection(b)

	def run(self, input_data, iterations):

		for i in range(iterations):

			for n in self.neurons:
				n.activate(input_data[random.randint(0, len(input_data)-1)])				

			for n in self.neurons:
				n.broadcast_intercon()
			
		test_result = []
		for n in self.neurons:
			test_result.append([n.activate(input_data[0]), n.activate(input_data[1])])
			
		return test_result
