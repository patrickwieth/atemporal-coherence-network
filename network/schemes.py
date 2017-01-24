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

	def set_mechanisms_by_dict(self, mechanisms_dict):
		input_processing = mechanisms_dict['input_processing']
		input_postprocessing = mechanisms_dict['input_postprocessing']
		activation_preprocessing = mechanisms_dict['activation_preprocessing']
		activation_processing = mechanisms_dict['activation_processing']
		activation_postprocessing = mechanisms_dict['activation_postprocessing']

		self.set_activation_phases([input_processing, input_postprocessing, activation_preprocessing, activation_processing, activation_postprocessing])
		self.set_broadcasting(mechanisms_dict['broadcast'], mechanisms_dict['receive'], mechanisms_dict['connect'])

	

### BASE SCHEME ###
base_scheme = scheme()

input_processing = [mechanisms.add_weights_when_input_too_long, mechanisms.sum_up_input]
input_postprocessing = [mechanisms.increase_weights_decrease_activation_on_weak_input,	mechanisms.define_unset_activation]
activation_preprocessing = [mechanisms.empty_actives, mechanisms.input_intensity_by_abs_diff]
activation_processing = [mechanisms.buff_activation_on_strong_input_nerv_on_weak_input]
activation_postprocessing = []

base_scheme.set_activation_phases([input_processing, input_postprocessing, activation_preprocessing, activation_processing, activation_postprocessing])
base_scheme.set_broadcasting([mechanisms.broadcast_intercon], [mechanisms.receive_intercon], [mechanisms.add_intercon])



# THESE SCHEMES ARE BULLSHIT:

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