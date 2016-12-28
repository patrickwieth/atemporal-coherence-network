import numpy as np
import math
from network import mechanisms

#########################################################################################################
# Schemes are sets of mechanisms			  															#
# Schemes offer a combination of mechanisms that bring a specific part of desired behavior to life		#
# A neuron can use a scheme, where the whole set of schemes makes up the behavior of a neuron 	        #
#########################################################################################################



def buff_weights_and_nerf_activation(neuron):
	mechanisms.nerf_activation(neuron)
	
	for idx in range(len(neuron.input_weights)):
		mechanisms.buff_weight(neuron, idx)


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

# interesting ideas


# kill dead weights forever, you and me!

# reshuffle (everything shitty for some time? reshuffle weights)

# backpropagation?