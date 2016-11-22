
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

