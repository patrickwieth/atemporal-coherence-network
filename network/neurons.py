import numpy as np
import random, math
from network import mechanisms

class connectron:
	def __init__(self, parameter):
		self.parameter = parameter
		self.activation = 0
		self.inputs = np.array([])
		self.input_weights = np.array([])
		self.input_mean = 0
		self.inhibited = np.array([])
		self.interconnected = []
		self.actives = []

	def set_interconnection(self, connected):
		self.interconnected.append(connected)

	def flush_inhibition(self):
		self.inhibited.fill(1)

	def receive_intercon(self, received):
		for i in received:
			# intercon scheme 2
			#self.inhibited[i] = 0.1

			#self.input_weights[i] -= self.parameter.intercon_diminishing * math.copysign(1, self.input_weights[i])
			self.input_weights[i] *= self.parameter.intercon_diminishing 

			# after a specific weight was diminished (because it was active on another connectron, all other weights get buffed)
			#buff = self.input_weights[i] * (1 - self.parameter.intercon_diminishing) / len(self.input_weights)
			#for j in self.input_weights:
			#	j += buff

	def broadcast_intercon(self):	
		# intercon scheme 2
		#return 1

		for i in self.interconnected:
			i.receive_intercon(self.actives)

			
	def activate(self, inputs):
		self.inputs = inputs

		while(len(inputs) > len(self.input_weights)):
			self.input_weights = np.append(self.input_weights, random.uniform(self.parameter.init_lower_weight, self.parameter.init_upper_weight))
			self.inhibited = np.append(self.inhibited, 1)

		input_sum = np.dot(inputs, self.input_weights)
		self.input_mean = input_sum / len(inputs)
		
		# without sufficient input, decrease activation, increase all weights, exit activate() 
		if(input_sum < self.parameter.threshold * self.activation):
			mechanisms.buff_weights_and_nerf_activation(self)
			return 0
	
		# if unset, set activation and exit activate()
		if(self.activation == 0):
			self.activation = input_sum
			return 0

		self.actives = []

		input_intensity = abs(input_sum) - abs(self.activation)

		if(input_intensity > 0):
			# strong input
			self.activation += self.parameter.activation_boost * math.copysign(1, input_sum)

			mechanisms.buff_or_nerf_depending_on_input(self)
			
			mechanisms.set_actives(self)
			
			
		else:
			# weak input
			self.activation *= 1 - self.parameter.activation_penalty

		return input_sum
