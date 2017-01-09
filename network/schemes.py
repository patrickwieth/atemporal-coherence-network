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
	



base_scheme = scheme()



def input_processing(neuron):
	mechanisms.add_weights_when_input_too_long(neuron)
	mechanisms.sum_up_input(neuron)

def input_postprocessing(neuron):
	mechanisms.increase_weights_decrease_activation_on_weak_input(neuron)
	mechanisms.define_unset_activation(neuron)

def activation_preprocessing(neuron):
	neuron.actives = []
	neuron.input_intensity = abs(neuron.input_sum) - abs(neuron.activation)

def activation_processing(neuron):
	if(neuron.input_intensity > 0):
		# strong input
		mechanisms.buff_activation(neuron)
		#self.activation += self.parameter.activation_buff * math.copysign(1, input_sum)

		mechanisms.buff_or_nerf_depending_on_input(neuron)
		mechanisms.set_actives(neuron)
	else:
		# weak input
		mechanisms.scale_down_activation(neuron)

def activation_postprocessing(neuron):
	return 0


base_scheme.set_activation_phases([input_processing, input_postprocessing, activation_preprocessing, activation_processing, activation_postprocessing])



# THESE SCHEMES ARE BULLSHIT:

'''
def buff_weights_and_nerf_activation(neuron):
	mechanisms.nerf_activation(neuron)
	
	for idx in range(len(neuron.input_weights)):
		mechanisms.buff_weight(neuron, idx)





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

# backpropagation?