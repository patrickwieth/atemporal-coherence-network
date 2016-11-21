#import numpy as np
import random, math




class supervised_connectron:
	def __init__(self):
		self.mean = 0
		self.threshold = 0.1
		self.input_weights = []

	def activate(self, inputs, supervision):
		self.mean = supervision

		while(len(inputs) > len(self.input_weights)):
			self.input_weights.append(random.randint(1,100)/100)

		input_sum = 0
		for i, value in enumerate(inputs):
			input_sum += value*self.input_weights[i]
		input_mean = input_sum / len(inputs)
		
		if(input_sum < self.threshold):
			return 0	

		diff = abs(input_sum) - abs(self.mean)

		if(diff > 0):
			for i, val in enumerate(inputs):
				if(abs(val*self.input_weights[i]) > abs(input_mean)):
					self.input_weights[i] -= 0.02 * math.copysign(1, self.input_weights[i])
				else:
					self.input_weights[i] *= 0.99
		else:
			for i, val in enumerate(inputs):
				if(abs(val*self.input_weights[i]) > abs(input_mean)):
					self.input_weights[i] += 0.02 * math.copysign(1, self.input_weights[i])
				else:
					self.input_weights[i] *= 0.99

		return input_sum


class connectron:
	def __init__(self):
		self.mean = 0
		self.threshold = 0.1
		self.input_weights = []
		self.interconnected = []
		self.actives = []

	def set_interconnection(self, connected):
		self.interconnected.append(connected)

	def receive_intercon(self, received):
		for i in received:
			self.input_weights[i] += -0.02 * math.copysign(1, self.input_weights[i])

	def broadcast_intercon(self):	
		for i in self.interconnected:
			i.receive_intercon(self.actives)

	def activate(self, inputs):
		while(len(inputs) > len(self.input_weights)):
			self.input_weights.append(random.randint(20,100)/100)

		input_sum = 0
		for i, value in enumerate(inputs):
			input_sum += value*self.input_weights[i]
		input_mean = input_sum / len(inputs)
		
		# only activate if there is enough input
		if(abs(input_sum) < self.threshold):
			return 0
	
		if(abs(self.mean) < self.threshold):
			self.mean = input_sum
			return 0

		self.actives = []
		diff = abs(input_sum) - abs(self.mean)
		#print("diff:", diff)

		if(diff > 0):
			# overshoot case
			self.mean += 0.01 * math.copysign(1, input_sum)

			for i, val in enumerate(inputs):
				if(abs(val*self.input_weights[i]) > abs(input_mean)):
					self.input_weights[i] -= 0.02 * math.copysign(1, self.input_weights[i])
					self.actives.append(i)
				else:
					self.input_weights[i] *= 0.99	
		else:
			# undershoot case
			self.mean *= 0.99
			
			for i, val in enumerate(inputs):
				if(abs(val*self.input_weights[i]) > abs(input_mean)):
					self.input_weights[i] += 0.02 * math.copysign(1, self.input_weights[i])
					self.actives.append(i)
				else:
					self.input_weights[i] *= 0.99

		return input_sum



def test_run1(iterations):
	a = connectron()
	b = connectron()

	a.set_interconnection(b)
	b.set_interconnection(a)

	input_vector_a = [0, 1, 0, 1]
	input_vector_b = [1, 0, 1, 0]	

	for i in range(iterations):
		rand_vector = [random.randrange(0,1), random.randrange(0,1), random.randrange(0,1), random.randrange(0,1)]

		activation = a.activate(rand_vector)
		activation = a.activate(input_vector_a)
		activation = a.activate(input_vector_b)

		activation = b.activate(rand_vector)
		activation = b.activate(input_vector_a)
		activation = b.activate(input_vector_b)

		a.broadcast_intercon()
		b.broadcast_intercon()
		
		
	print(a.activate(input_vector_a), a.activate(input_vector_b))
	print(b.activate(input_vector_a), b.activate(input_vector_b))
	

test_run1(1000)
	

def test_run2(iterations):
	a = supervised_connectron()	
	b = supervised_connectron()	

	input_vector_a = [0, 1, 0, 1]
	input_vector_b = [1, 0, 1, 0]

	for i in range(iterations):
		rand_vector = [random.randrange(0,1), random.randrange(0,1), random.randrange(0,1), random.randrange(0,1)]

		activation = b.activate(rand_vector, 0)
		activation = a.activate(rand_vector, 0)
		activation = b.activate(input_vector_a, 0)
		activation = a.activate(input_vector_a, 1)
		activation = b.activate(input_vector_b, 1)
		activation = a.activate(input_vector_b, 0)
		#print(activation, a.input_weights)

	print(a.activate(input_vector_a, 0))
	print(a.activate(input_vector_b, 0))
	print(b.activate(input_vector_a, 0))
	print(b.activate(input_vector_b, 0))



