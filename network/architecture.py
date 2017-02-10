import random
from network import neurons
from network import schemes

'''
class parameter:
	def __init__(self, threshold, init_weight_bounds, weight_ajdustors, activation_adjustors, intercon_diminishing):
		self.threshold = threshold
		self.init_lower_weight = init_weight_bounds[0]
		self.init_upper_weight = init_weight_bounds[1]
		self.weight_buff = weight_ajdustors[0]
		self.weight_nerf = weight_ajdustors[1]
		self.activation_buff = activation_adjustors[0]
		self.activation_nerf = activation_adjustors[1]
		self.activation_scale_down = activation_adjustors[2]
		#self.activation_scale_up = activation_adjustors[3]
		self.intercon_diminishing = intercon_diminishing

	def get_dict(self):
		return {
		'threshold': self.threshold,
		'init_lower_weight': self.init_lower_weight,
		'init_upper_weight': self.init_upper_weight,
		'weight_buff': self.weight_buff,
		'weight_nerf': self.weight_nerf,
		'activation_buff': self.activation_buff,
		'activation_nerf': self.activation_nerf,
		'activation_scale_down': self.activation_scale_down,
		'intercon_diminishing': self.intercon_diminishing
		}
'''

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

'''
def flat_array_to_parameter(array):
	return parameter(array[0], [array[1], array[2]], [array[3], array[4]], [array[5], array[6], array[7]], array[8])
'''
