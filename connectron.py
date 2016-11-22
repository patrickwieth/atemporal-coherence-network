#import numpy as np
import random, math

class network:
	def __init__(self, size):
		self.neurons = []

		for i in range(size):
			self.neurons.append(connectron())

		for i, a in enumerate(self.neurons):
			for j, b in enumerate(self.neurons):
				if(i != j):
					a.set_interconnection(b)

	def run(self, input_data, iterations):

		for i in range(iterations):

			for n in self.neurons:
				n.activate(input_data[random.randint(0, len(input_data)-1)])				

			for n in self.neurons:
				n.broadcast_intercon()
			
		for n in self.neurons:
			print("{:f} {:f}".format(n.activate(input_data[0]), n.activate(input_data[1])))



class supervised_connectron:
	def __init__(self):
		self.activation = 0
		self.threshold = 0.1
		self.input_weights = []

	def activate(self, inputs, supervision):
		self.activation = supervision

		while(len(inputs) > len(self.input_weights)):
			self.input_weights.append(random.randint(10,100)/100)

		input_sum = 0
		for i, value in enumerate(inputs):
			input_sum += value*self.input_weights[i]
		input_mean = input_sum / len(inputs)
		
		if(input_sum < self.threshold):
			return 0	

		diff = abs(input_sum) - abs(self.activation)

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
		self.activation = 0
		self.threshold = 0.2
		self.input_weights = []
		self.interconnected = []
		self.actives = []

	def set_interconnection(self, connected):
		self.interconnected.append(connected)

	def receive_intercon(self, received):
		for i in received:
			#self.input_weights[i] += -0.02 * math.copysign(1, self.input_weights[i])
			self.input_weights[i] *= 0.99 #* math.copysign(1, self.input_weights[i])

	def broadcast_intercon(self):	
		for i in self.interconnected:
			i.receive_intercon(self.actives)

	def activate(self, inputs):
		while(len(inputs) > len(self.input_weights)):
			self.input_weights.append(random.randint(50,200)/100)

		input_sum = 0
		for i, value in enumerate(inputs):
			input_sum += value*self.input_weights[i]
		input_mean = input_sum / len(inputs)
		
		# without sufficient input, increase all weights, decrease activation
		if(input_sum < self.threshold * self.activation):
			self.activation * 0.99
			for i in self.input_weights:
				i += 0.02 * math.copysign(1, i)
			return 0
	
		# if unset, set activation
		if(self.activation == 0):
			self.activation = input_sum
			return 0

		self.actives = []
		diff = abs(input_sum) - abs(self.activation)

		if(diff > 0):
			# strong input
			self.activation += 0.02 * math.copysign(1, input_sum)

			for i, val in enumerate(inputs):
				# decrease unimportant inputs
				if(abs(val*self.input_weights[i]) < abs(input_mean)):
					self.input_weights[i] -= 0.02 * math.copysign(1, self.input_weights[i])
				else:					
					#self.input_weights[i] += 0.02 * math.copysign(1, self.input_weights[i])
					#self.input_weights[i] *= 0.99	
					self.actives.append(i)
		else:
			# weak input
			self.activation *= 0.99

		return input_sum


input_a = [0, 1, 0, 1]
input_b = [1, 0, 1, 0]

def rand_input():
	return [random.randrange(0,1), random.randrange(0,1), random.randrange(0,1), random.randrange(0,1)]

data = []

data.append(input_a)
data.append(input_b)

for i in range(100):
	pick = random.randint(0,2)
	if(pick == 0):
		data.append(input_a)
	elif(pick == 1):
		data.append(input_b)
	else:
		data.append(rand_input())

ne_is_ok = network(2)

ne_is_ok.run(data, 1000)



def test_run1(iterations):
	a = connectron()
	b = connectron()

	a.set_interconnection(b)
	b.set_interconnection(a)

	input_vector_a = [0, 0.1, 0, 0.1]
	input_vector_b = [0.1, 0, 0.1, 0]

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
		

	print("{:f} {:f}".format(a.activate(input_vector_a), a.activate(input_vector_b)))
	print("{:f} {:f}".format(b.activate(input_vector_a), b.activate(input_vector_b)))
	

#test_run1(2000)
	

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



