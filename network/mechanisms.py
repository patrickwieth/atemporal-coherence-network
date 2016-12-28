import numpy as np
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

# weights

def scale_up_weight(neuron, idx, modulation):
	neuron.input_weights[idx] *= 1 + (neuron.parameter.weight_scale_up * modulation)

def scale_down_weight(neuron, idx, modulation):
	neuron.input_weights[idx] *= 1 - (neuron.parameter.weight_scale_down * modulation)

def buff_weight(neuron, idx):
	neuron.input_weights[idx] += math.copysign(neuron.parameter.weight_buff, neuron.input_weights[idx])

def nerf_weight(neuron, idx):
	neuron.input_weights[idx] -= math.copysign(neuron.parameter.weight_nerf, neuron.input_weights[idx])	

# input

def sum_up_input(neuron):
	neuron.input_sum = np.dot(neuron.inputs, neuron.input_weights)
	neuron.input_mean = neuron.input_sum / len(neuron.inputs)

def sum_up_uninhibited_input(neuron):
	effective_weights = neuron.input_weights * neuron.inhibited

	neuron.input_sum = np.dot(neuron.inputs, effective_weights)
	neuron.input_mean = neuron.input_sum / len(neuron.inputs)

def flush_inhibition(neuron):
	neuron.inhibited.fill(1)

# inhibtion

def set_actives(neuron):
	for i, val in enumerate(neuron.inputs):
		if(abs(val*neuron.input_weights[i]) > abs(neuron.input_mean)):
			neuron.actives.append(i)
			






