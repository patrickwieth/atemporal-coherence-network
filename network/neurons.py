import numpy as np
import random, math

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

		
		self.activation_phases = [[],[],[],[],[]]

		self.broadcast_fns = []
		self.connect_fns = []
		self.receive_fns = []


		for s_idx, s in enumerate(schemes):
			for activation_idx, fns in enumerate(s.activation_phases):
				for fn in fns:
					self.activation_phases[activation_idx].append(fn)

			self.broadcast_fns = list(s.broadcast)
			self.receive_fns = list(s.receive)
			self.connect_fns = list(s.connect)


	def add_interconnection(self, connected):
		for fn in self.connect_fns:
			fn(self, connected)
	
	def receive_intercon(self, received):
		for fn in self.receive_fns:
			fn(self, received)

	def broadcast(self):
		for fn in self.broadcast_fns:
			fn(self)
				
	def activate(self, inputs):
		self.stop_flag = False

		self.inputs = inputs

		for acts in self.activation_phases:
			for fn in acts:
				fn(self)
			if(self.stop_flag): return 0

		return self.input_sum
