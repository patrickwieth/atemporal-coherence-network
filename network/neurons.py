#import numpy as np
import random, math


class supervised_connectron:
	def __init__(self):
		self.activation = 0
		self.threshold = 0.1
		self.input_weights = []

	def activate(self, inputs, supervision):
		self.activation = supervision

		while(len(inputs) > len(self.input_weights)):
			self.input_weights.append(random.randint(10,100)/100)

		input_sum = 0
		for i, value in enumerate(inputs):
			input_sum += value*self.input_weights[i]
		input_mean = input_sum / len(inputs)
		
		if(input_sum < self.threshold):
			return 0	

		diff = abs(input_sum) - abs(self.activation)

		if(diff > 0):
			for i, val in enumerate(inputs):
				if(abs(val*self.input_weights[i]) > abs(input_mean)):
					self.input_weights[i] -= 0.02 * math.copysign(1, self.input_weights[i])
				else:
					self.input_weights[i] *= 0.99
		else:
			for i, val in enumerate(inputs):
				if(abs(val*self.input_weights[i]) > abs(input_mean)):
					self.input_weights[i] += 0.02 * math.copysign(1, self.input_weights[i])
				else:
					self.input_weights[i] *= 0.99

		return input_sum


class connectron:
	def __init__(self, parameter):
		self.parameter = parameter
		self.activation = 0
		self.threshold = 0.2
		self.input_weights = []
		self.interconnected = []
		self.actives = []

	def set_interconnection(self, connected):
		self.interconnected.append(connected)

	def receive_intercon(self, received):
		for i in received:
			#self.input_weights[i] += -0.02 * math.copysign(1, self.input_weights[i])
			self.input_weights[i] *= 0.99 #* math.copysign(1, self.input_weights[i])

	def broadcast_intercon(self):	
		for i in self.interconnected:
			i.receive_intercon(self.actives)

	def activate(self, inputs):
		while(len(inputs) > len(self.input_weights)):
			self.input_weights.append(random.uniform(self.parameter.init_lower_weight, self.parameter.init_upper_weight))
			

		input_sum = 0
		for i, value in enumerate(inputs):
			input_sum += value*self.input_weights[i]
		input_mean = input_sum / len(inputs)
		
		# without sufficient input, increase all weights, decrease activation
		if(input_sum < self.threshold * self.activation):
			self.activation * self.parameter.activation_diminishing
			for i in self.input_weights:
				i += 0.02 * math.copysign(1, i)
				#i += self.parameter.weight_boost * math.copysign(1, i)
			return 0
	
		# if unset, set activation
		if(self.activation == 0):
			self.activation = input_sum
			return 0

		self.actives = []
		diff = abs(input_sum) - abs(self.activation)

		if(diff > 0):
			# strong input
			self.activation += 0.02 * math.copysign(1, input_sum)

			for i, val in enumerate(inputs):
				# decrease unimportant inputs
				if(abs(val*self.input_weights[i]) < abs(input_mean)):
					self.input_weights[i] -= 0.02 * math.copysign(1, self.input_weights[i])
				else:					
					#self.input_weights[i] += 0.02 * math.copysign(1, self.input_weights[i])
					#self.input_weights[i] *= 0.99	
					self.actives.append(i)
		else:
			# weak input
			self.activation *= 0.99

		return input_sum

