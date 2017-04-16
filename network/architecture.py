import random
from network import neurons
from network import schemes

class topology:
	def __init__(self, size):
		self.size = size

class instance:
	def __init__(self, topology, behavior, parameter):

		self.neurons = []

		for i in range(topology.size):
			self.neurons.append(neurons.connectron(parameter, behavior))

		for i, a in enumerate(self.neurons):
			for j, b in enumerate(self.neurons):
				if(i != j):
					a.add_interconnection(b)

	def run(self, input_data, iterations):
		for i in range(iterations):
			for n in self.neurons:
				n.activate(input_data[random.randint(0, len(input_data)-1)])				

			for n in self.neurons:
				n.broadcast()
				
	def test(self, input_data, iterations):
		test_result = []
		for n in self.neurons:
			neuron_activation = []
			for i in range(iterations):
				neuron_activation.append(n.activate(input_data[i]))

			test_result.append(neuron_activation)

		return test_result
