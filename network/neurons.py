import numpy as np
import random, math
from network import mechanisms
from network import schemes

class connectron:
	def __init__(self, parameter, schemes):
		self.parameter = parameter
		self.activation = 0
		self.inputs = np.array([])
		self.input_weights = np.array([])
		self.input_mean = 0
		self.input_sum = 0
		self.input_intensity = 0
		self.inhibited = np.array([])
		self.interconnected = []
		self.actives = []
		self.stop_flag = False

		# ACTIVATION

		# eigentlich sollte hier jede Funktion aus jeder Phase an jede Phase angehängt werden...
		self.activation_phases = schemes.activation_phases

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
			if(self.input_intensity > 0):
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

		self.activation_phases.append(input_processing)
		self.activation_phases.append(input_postprocessing)
		self.activation_phases.append(activation_preprocessing)
		self.activation_phases.append(activation_processing)
		self.activation_phases.append(activation_postprocessing)

		# INTERCON BROADCASTING

		self.broadcast_intercon = lambda: mechanisms.broadcast_intercon(self)
		self.receive_intercon = lambda received: mechanisms.receive_intercon(self, received)
		self.add_interconnection = lambda connected: mechanisms.add_intercon(self, connected)

	
	def broadcast(self):
		mechanisms.broadcast_intercon(self)
				
	def activate(self, inputs):
		self.stop_flag = False

		# phase 1: INPUT PROCESSING
		self.inputs = inputs
		self.activation_phases[0](self)
		if(self.stop_flag): return 0

		# phase 2: INPUT POSTPROCESSING
		self.activation_phases[1](self)
		if(self.stop_flag): return 0

		# phase 3: ACTIVATION PREPROCESSING
		self.activation_phases[2](self)
		if(self.stop_flag): return 0

		# phase 4: ACTIVATION PROCESSING
		self.activation_phases[3](self)
		if(self.stop_flag): return 0

		# phase 5: ACTIVATION POSTPROCESSING
		self.activation_phases[3](self)
		if(self.stop_flag): return 0

		return self.input_sum
