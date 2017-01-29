import numpy as np
import math
import copy
from network import mechanisms

#########################################################################################################
# Schemes are sets of mechanisms			  															#
# Schemes offer a combination of mechanisms that bring a specific part of desired behavior to life		#
# A neuron can use a scheme, where the whole set of schemes makes up the behavior of a neuron 	        #
#########################################################################################################

class scheme:
	def __init__(self):
		self.parameters = {}
		self.activation_phases = []
		self.broadcast = []
		self.receive = []
		self.connect =[]
		
	def set_activation_phases(self, args):
		self.activation_phases = args

	def set_broadcasting(self, send, receive, connect):
		self.broadcast = send
		self.connect = connect
		self.receive = receive

	def integrate_schemes(self, schemes):
		for x in schemes:
			self.parameters.update(x.parameters)

			self.broadcast += x.broadcast
			self.receive += x.receive
			self.connect += x.connect

			for idx, phase in enumerate(self.activation_phases):
				phase += x.activation_phases[idx]

	def set_mechanisms_by_list(self, mechanisms_list):
		mechs = {
		'input_processing': [],
		'input_postprocessing': [],
		'activation_preprocessing': [],
		'activation_processing': [],
		'activation_postprocessing': [],
		'broadcast': [],
		'receive': [],
		'connect': []
		}

		for x in mechanisms_list:
			mechs[x.phase] += [x.fn]

			for y in x.parameters:
				self.parameters[y['name']] = [y['lower_limit'], y['upper_limit']]

		self.set_activation_phases([mechs['input_processing'], mechs['input_postprocessing'], mechs['activation_preprocessing'], mechs['activation_processing'], mechs['activation_postprocessing']])
		self.set_broadcasting(mechs['broadcast'], mechs['receive'], mechs['connect'])



### BASE SCHEME ###
base_scheme = scheme()

base_scheme.set_mechanisms_by_list([mechanisms.add_weights_when_input_too_long, 
									mechanisms.sum_up_input,
									mechanisms.increase_weights_decrease_activation_on_weak_input, 
									mechanisms.define_unset_activation,
									mechanisms.empty_actives, 
									mechanisms.input_intensity_by_abs_diff,
									mechanisms.buff_activation_on_strong_input_nerf_on_weak_input,
									mechanisms.broadcast_intercon, 
									mechanisms.receive_intercon, 
									mechanisms.add_intercon])


# THESE SCHEMES ARE BULLSHIT:  (think about turning into mechanisms)

'''
def set_actives(neuron):
	for i, val in enumerate(neuron.inputs):

		if(abs(val*neuron.input_weights[i]) < abs(neuron.input_mean)):
			4
			# decrease unimportant inputs
			#self.input_weights[i] -= self.parameter.weight_nerf * math.copysign(1, self.input_weights[i])
		
		else:
			# append important inputs to actives
			neuron.actives.append(i)
			#self.input_weights[i] += self.parameter.weight_buff * math.copysign(1, i)
			
			# intercon scheme 2
			#for j in self.interconnected:
			#	j.receive_intercon([i])


# interconnection

def receive_intercon(neuron, received):
	for idx in received:
		mechanisms.scale_down_weight(neuron, idx, self.parameter.intercon_diminishing)

		# after a specific weight was diminished (because it was active on another connectron, all other weights get buffed)
		#buff = self.input_weights[i] * (1 - self.parameter.intercon_diminishing) / len(self.input_weights)
		#for j in self.input_weights:
		#	j += buff

def receive_intercon2(neuron, received):
	for i in received:
		
		self.inhibited[i] = 0.1

		# after a specific weight was diminished (because it was active on another connectron, all other weights get buffed)
		#buff = self.input_weights[i] * (1 - self.parameter.intercon_diminishing) / len(self.input_weights)
		#for j in self.input_weights:
		#	j += buff

def broadcast_intercon(self):	
	# intercon scheme 2
	#return 1

	for i in self.interconnected:
		i.receive_intercon(self.actives)
'''

# interesting ideas

# remove dead weights
# reshuffle (everything shitty for some time? reshuffle weights)
# backpropagation??