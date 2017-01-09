import numpy as np
import random
import math


#########################################################################################################
# Mechanisms are building blocks of neurons  															#
# A scheme can use a mechanism, where the whole set of mechanisms makes up the behavior of a neuron 	#
#########################################################################################################

# activation

def scale_up_activation(neuron):
	neuron.activation *= 1 + neuron.parameter.activation_scale_up

def scale_down_activation(neuron):
	neuron.activation *= 1 - neuron.parameter.activation_scale_down

def buff_activation(neuron):
	neuron.activation += math.copysign(neuron.parameter.activation_buff, neuron.activation)

def nerf_activation(neuron):
	neuron.activation -= math.copysign(neuron.parameter.activation_nerf, neuron.activation)	

def define_unset_activation(neuron):
	if(neuron.activation == 0):
		neuron.activation = neuron.input_sum
		neuron.stop_flag = True
		return 0

# weights

def scale_up_weight(neuron, idx, modulation):
	neuron.input_weights[idx] *= 1 + (neuron.parameter.weight_scale_up * modulation)

def scale_down_weight(neuron, idx, modulation):
	neuron.input_weights[idx] *= 1 - (neuron.parameter.weight_scale_down * modulation)

def buff_weight(neuron, idx):
	neuron.input_weights[idx] += math.copysign(neuron.parameter.weight_buff, neuron.input_weights[idx])

def nerf_weight(neuron, idx):
	neuron.input_weights[idx] -= math.copysign(neuron.parameter.weight_nerf, neuron.input_weights[idx])	

def buff_weights_and_nerf_activation(neuron):
	nerf_activation(neuron)
	
	for idx in range(len(neuron.input_weights)):
		buff_weight(neuron, idx)


# input

def sum_up_input(neuron):
	neuron.input_sum = np.dot(neuron.inputs, neuron.input_weights)
	neuron.input_mean = neuron.input_sum / len(neuron.inputs)

def sum_up_uninhibited_input(neuron):
	effective_weights = neuron.input_weights * neuron.inhibited

	neuron.input_sum = np.dot(neuron.inputs, effective_weights)
	neuron.input_mean = neuron.input_sum / len(neuron.inputs)

def add_weights_when_input_too_long(neuron):
	while(len(neuron.inputs) > len(neuron.input_weights)):
		neuron.input_weights = np.append(neuron.input_weights, random.uniform(neuron.parameter.init_lower_weight, neuron.parameter.init_upper_weight))
		neuron.inhibited = np.append(neuron.inhibited, 1)

def increase_weights_decrease_activation_on_weak_input(neuron):
	if(neuron.input_sum < neuron.parameter.threshold * neuron.activation):
		buff_weights_and_nerf_activation(neuron)
		neuron.stop_flag = True
		return 0

def buff_or_nerf_depending_on_input(neuron):
	
	def buff_or_nerf(a):
		if abs(a[0]*a[1]) > abs(neuron.input_mean):
			#self.actives.append(i)
			a[0] += math.copysign(neuron.parameter.weight_buff, a[0])  
		else:
			a[0] -= math.copysign(neuron.parameter.weight_nerf, a[0])

		return a

	inputyes = np.array([neuron.input_weights, neuron.inputs]).T
	#print(inputyes)
	inputyes = np.array(list(map(buff_or_nerf, inputyes)))	

			
	neuron.input_weights = inputyes.T[0]


def flush_inhibition(neuron):
	neuron.inhibited.fill(1)

# inhibtion

def set_actives(neuron):
	for i, val in enumerate(neuron.inputs):
		if(abs(val*neuron.input_weights[i]) > abs(neuron.input_mean)):
			neuron.actives.append(i)
			
# interconnection

def broadcast_intercon(neuron):	
	for i in neuron.interconnected:
		i.receive_intercon(neuron.actives)

def receive_intercon(neuron, received):
	for i in received:
		neuron.input_weights[i] *= neuron.parameter.intercon_diminishing 

def add_intercon(neuron, connected):
	neuron.interconnected.append(connected)

