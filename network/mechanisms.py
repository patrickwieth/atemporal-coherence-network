import numpy as np
import random
import math

#########################################################################################################
# Mechanisms are building blocks of neurons  															#
# A scheme can use a mechanism, where the whole set of mechanisms makes up the behavior of a neuron 	#
#########################################################################################################

class mechanism:
	def __init__(self, name, fn, phase, parameters):
		self.name = name
		self.fn = fn
		self.phase = phase
		self.parameters = parameters

	def inherit_parameters(self, params):
		self.parameters += params

	def pass_parameters(self):
		return self.parameters

# activation

def scale_up_activation_fn(neuron):	# currently unused!
	neuron.activation *= 1 + neuron.parameter['activation_scale_up']

scale_up_activation = mechanism('scale_up_activation', scale_up_activation_fn, 'none',
								[{'name': 'activation_scale_up', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1}])


def scale_down_activation_fn(neuron):
	neuron.activation *= 1 - neuron.parameter['activation_scale_down']

scale_down_activation = mechanism('scale_down_activation', scale_down_activation_fn, 'none',
								  [{'name': 'activation_scale_down', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1}])


def buff_activation_fn(neuron):
	neuron.activation += math.copysign(neuron.parameter['activation_buff'], neuron.activation)

buff_activation = mechanism('buff_activation', buff_activation_fn, 'none',
							[{'name': 'activation_buff', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1}])


def nerf_activation_fn(neuron):
	neuron.activation -= math.copysign(neuron.parameter['activation_nerf'], neuron.activation)

nerf_activation = mechanism('nerf_activation', nerf_activation_fn, 'none',
							[{'name': 'activation_nerf', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1}])


def define_unset_activation_fn(neuron):
	if(neuron.activation == 0):
		neuron.activation = neuron.input_sum
		neuron.stop_flag = True
		return 0

define_unset_activation = mechanism('define_unset_activation', define_unset_activation_fn, 'input_postprocessing', [])

# weights

def scale_up_weight_fn(neuron, idx, modulation):	# currently unused!
	neuron.input_weights[idx] *= 1 + (neuron.parameter['weight_scale_up'] * modulation)

scale_up_weight = mechanism('scale_up_weight', scale_up_weight_fn,  'none',
							[{'name': 'weight_scale_up', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1}])


def scale_down_weight_fn(neuron, idx, modulation):	# currently unused!
	neuron.input_weights[idx] *= 1 - (neuron.parameter['weight_scale_down'] * modulation)

scale_down_weight = mechanism('scale_down_weight', scale_down_weight_fn,  'none',
							  [{'name': 'weight_scale_down', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1}])


def buff_weight_fn(neuron, idx):
	neuron.input_weights[idx] += math.copysign(neuron.parameter['weight_buff'], neuron.input_weights[idx])

buff_weight = mechanism('buff_weight', buff_weight_fn, 'none',
						[{'name': 'weight_buff', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1}])


def nerf_weight_fn(neuron, idx):
	neuron.input_weights[idx] -= math.copysign(neuron.parameter['weight_nerf'], neuron.input_weights[idx])	

nerf_weight = mechanism('nerf_weight', nerf_weight_fn, 'none',
						[{'name': 'weight_nerf', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1}])


def buff_weights_and_nerf_activation_fn(neuron):
	nerf_activation.fn(neuron)
	
	for idx in range(len(neuron.input_weights)):
		buff_weight.fn(neuron, idx)

buff_weights_and_nerf_activation = mechanism('buff_weights_and_nerf_activation', buff_weights_and_nerf_activation_fn,  'none', [])
buff_weights_and_nerf_activation.inherit_parameters(buff_weight.pass_parameters())
buff_weights_and_nerf_activation.inherit_parameters(nerf_activation.pass_parameters())

# input

def sum_up_input_fn(neuron):
	neuron.input_sum = np.dot(neuron.inputs, neuron.input_weights)
	neuron.input_mean = neuron.input_sum / len(neuron.inputs)

sum_up_input = mechanism('sum_up_input', sum_up_input_fn, 'input_processing', [])


def sum_up_uninhibited_input_fn(neuron): # currently not used!
	effective_weights = neuron.input_weights * neuron.inhibited

	neuron.input_sum = np.dot(neuron.inputs, effective_weights)
	neuron.input_mean = neuron.input_sum / len(neuron.inputs)

sum_up_uninhibited_input = mechanism('sum_up_uninhibited_input', sum_up_uninhibited_input_fn, 'input_processing' ,[])


def add_weights_when_input_too_long_fn(neuron):
	while(len(neuron.inputs) > len(neuron.input_weights)):
		neuron.input_weights = np.append(neuron.input_weights, random.uniform(neuron.parameter['init_lower_weight'], neuron.parameter['init_upper_weight']))
		neuron.inhibited = np.append(neuron.inhibited, 1)

add_weights_when_input_too_long = mechanism('add_weights_when_input_too_long', add_weights_when_input_too_long_fn, 'input_processing',
											[{'name': 'init_lower_weight', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1},
											 {'name': 'init_upper_weight', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1}])


def increase_weights_decrease_activation_on_weak_input_fn(neuron):
	if(neuron.input_sum < neuron.parameter['threshold'] * neuron.activation):
		buff_weights_and_nerf_activation.fn(neuron)
		neuron.stop_flag = True
		return 0

increase_weights_decrease_activation_on_weak_input = mechanism('increase_weights_decrease_activation_on_weak_input', increase_weights_decrease_activation_on_weak_input_fn, 'input_postprocessing',
															   [{'name': 'threshold', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1}])
increase_weights_decrease_activation_on_weak_input.inherit_parameters(buff_weights_and_nerf_activation.pass_parameters())


def scale_weights_depending_on_input_fn(neuron):

	def scale_up_or_down(a):
		if abs(a[0]*a[1]) > abs(neuron.input_mean):
			a[0] *= 1 + neuron.parameter['weight_scale_up'] 
		else:
			a[0] *= 1 - neuron.parameter['weight_scale_down']
		return a

	# HIER WURDE SCHEISSE PROGRAMMIERT, SPEEDUP EXTRAHIEREN ufunc?
	inputyes = np.array([neuron.input_weights, neuron.inputs]).T
	#print(inputyes)
	inputyes = np.array(list(map(scale_up_or_down, inputyes)))	
	
	neuron.input_weights = inputyes.T[0]

scale_weights_depending_on_input = mechanism('scale_weights_depending_on_input', scale_weights_depending_on_input_fn, 'none',
											[{'name': 'weight_scale_up', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1},
											 {'name': 'weight_scale_down', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1}])


def buff_or_nerf_depending_on_input_fn(neuron):
	
	def buff_or_nerf(a):
		if abs(a[0]*a[1]) > abs(neuron.input_mean):
			a[0] += math.copysign(neuron.parameter['weight_buff'], a[0])  
		else:
			a[0] -= math.copysign(neuron.parameter['weight_nerf'], a[0])
		return a

	# HIER WURDE SCHEISSE PROGRAMMIERT, SPEEDUP EXTRAHIEREN ufunc?
	inputyes = np.array([neuron.input_weights, neuron.inputs]).T
	#print(inputyes)
	inputyes = np.array(list(map(buff_or_nerf, inputyes)))	
	
	neuron.input_weights = inputyes.T[0]

buff_or_nerf_depending_on_input = mechanism('buff_or_nerf_depending_on_input', buff_or_nerf_depending_on_input_fn, 'none',
											[{'name': 'weight_buff', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1},
											 {'name': 'weight_nerf', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1}])


def input_intensity_by_abs_diff_fn(neuron):
	neuron.input_intensity = abs(neuron.input_sum) - abs(neuron.activation)

input_intensity_by_abs_diff = mechanism('input_intensity_by_abs_diff', input_intensity_by_abs_diff_fn, 'activation_preprocessing', [])


# inhibtion

def flush_inhibition_fn(neuron):
	neuron.inhibited.fill(1)

flush_inhibition = mechanism('flush_inhibition', flush_inhibition_fn, 'none', []) #change 'none' to something else, when implemented


def empty_actives_fn(neuron):
	neuron.actives = []

empty_actives = mechanism('empty_actives', empty_actives_fn, 'activation_preprocessing', [])


def set_actives_fn(neuron):
	for i, val in enumerate(neuron.inputs):
		if(abs(val*neuron.input_weights[i]) > abs(neuron.input_mean)):
			neuron.actives.append(i)

set_actives = mechanism('set_actives', set_actives_fn, 'none', [])

			
# big mechanisms

def buff_activation_on_strong_input_nerf_on_weak_input_fn(neuron):
	if(neuron.input_intensity > 0):
		# strong input
		buff_activation.fn(neuron)
		buff_or_nerf_depending_on_input.fn(neuron)
		set_actives.fn(neuron)
	else:
		# weak input
		scale_down_activation.fn(neuron)

buff_activation_on_strong_input_nerf_on_weak_input = mechanism('buff_activation_on_strong_input_nerf_on_weak_input', buff_activation_on_strong_input_nerf_on_weak_input_fn, 'activation_processing', [])
buff_activation_on_strong_input_nerf_on_weak_input.inherit_parameters(buff_or_nerf_depending_on_input.pass_parameters())
buff_activation_on_strong_input_nerf_on_weak_input.inherit_parameters(scale_down_activation.pass_parameters())
buff_activation_on_strong_input_nerf_on_weak_input.inherit_parameters(buff_activation.pass_parameters())
buff_activation_on_strong_input_nerf_on_weak_input.inherit_parameters(set_actives.pass_parameters())


def scale_weights_on_strong_input_scale_down_activation_on_weak_input_fn(neuron):
	if(neuron.input_intensity > 0):
		# strong input
		buff_activation.fn(neuron)
		scale_weights_depending_on_input.fn(neuron)
		set_actives.fn(neuron)
	else:
		# weak input
		scale_down_activation.fn(neuron)

scale_weights_on_strong_input_scale_down_activation_on_weak_input = mechanism('scale_weights_on_strong_input_scale_down_activation_on_weak_input', scale_weights_on_strong_input_scale_down_activation_on_weak_input_fn, 'activation_processing', [])
scale_weights_on_strong_input_scale_down_activation_on_weak_input.inherit_parameters(scale_weights_depending_on_input.pass_parameters())
scale_weights_on_strong_input_scale_down_activation_on_weak_input.inherit_parameters(scale_down_activation.pass_parameters())
scale_weights_on_strong_input_scale_down_activation_on_weak_input.inherit_parameters(buff_activation.pass_parameters())
scale_weights_on_strong_input_scale_down_activation_on_weak_input.inherit_parameters(set_actives.pass_parameters())


# interconnection

def broadcast_intercon_fn(neuron):	
	for i in neuron.interconnected:
		i.receive_intercon(neuron.actives)

broadcast_intercon = mechanism('broadcast_intercon', broadcast_intercon_fn, 'broadcast', [])		


def receive_intercon_fn(neuron, received):
	for i in received:
		neuron.input_weights[i] *= neuron.parameter['intercon_diminishing']

receive_intercon = mechanism('receive_intercon', receive_intercon_fn, 'receive',	
							 [{'name': 'intercon_diminishing', 'type': np.float, 'lower_limit': 0, 'upper_limit': 1}])


def add_intercon_fn(neuron, connected):
	neuron.interconnected.append(connected)

add_intercon = mechanism('add_intercon', add_intercon_fn, 'connect', [])


def do_nothing_fn(neuron):
	1

do_nothing = mechanism('do_nothing', do_nothing_fn, 'activation_postprocessing', [])

