import numpy as np
import math

#########################################################################################################
# Mechanisms are building blocks of neurons  															#
# a neuron can use a mechanism, where the whole set of mechanisms makes up the behavior of a neurons 	#
#########################################################################################################

def buff_weights_and_nerf_activation(neuron):
	neuron.activation * neuron.parameter.activation_diminishing
	neuron.input_weights += neuron.parameter.weight_boost  #* math.copysign(1, i)

def buff_or_nerf_depending_on_input(neuron):
	

	def buff_or_nerf(a):
		if abs(a[0]*a[1]) > abs(neuron.input_mean):
			#self.actives.append(i)
			a[0] += math.copysign(neuron.parameter.weight_boost, a[0])  
		else:
			a[0] -= math.copysign(neuron.parameter.weight_penalty, a[0])

		return a


	inputyes = np.array([neuron.input_weights, neuron.inputs]).T
	#print(inputyes)
	inputyes = np.array(list(map(buff_or_nerf, inputyes)))	

			
	neuron.input_weights = inputyes.T[0]


def set_actives(neuron):
	for i, val in enumerate(neuron.inputs):

		if(abs(val*neuron.input_weights[i]) < abs(neuron.input_mean)):
			4
			# decrease unimportant inputs
			#self.input_weights[i] -= self.parameter.weight_penalty * math.copysign(1, self.input_weights[i])
		
		else:
			# append important inputs to actives
			neuron.actives.append(i)
			#self.input_weights[i] += self.parameter.weight_boost * math.copysign(1, i)
			
			# intercon scheme 2
			#for j in self.interconnected:
			#	j.receive_intercon([i])


