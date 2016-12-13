import random
import numpy as np

# creates sample data of recurring patterns and plain noise, the recurring patterns can also be given noise
# data always starts with patterns and is followed by a corresponding amount of random data
def patterns(number_of_patterns, percentage_of_random_data, noise_on_patterns):

	def rand_input():
		return [random.uniform(0, 1) for _ in range(number_of_patterns)]

	def add_noise(data):
		f = lambda x : x + random.uniform(-x * noise_on_patterns, x * noise_on_patterns)
		return(list(map(f, data)))

	def push_pattern(sample, available_patterns):
		pick = random.randint(0, number_of_patterns)
		sample.append(add_noise(available_patterns[pick]))

	def push_random(sample):
		sample.append(rand_input())

	def push_datum(sample, available_patterns):
		pattern_or_random = random.uniform(0, 1)

		if(pattern_or_random > percentage_of_random_data):
			push_datum(sample, available_patterns)
		else:
			push_random(sample)

	def create_input_patterns(dimension, number_of_patterns):
		patterns = []

		if(number_of_patterns > 2**dimension):
			print("Not possible to create", number_of_patterns, "number of patterns on", dimension, "dimensions... creating the maximal number")
			number_of_patterns = 2**dimension

		for i in range(number_of_patterns):
			binary_number = bin(i)[2:].zfill(dimension)
			patterns.append(list(map(lambda x: int(x), binary_number)))

		return patterns

	# strip off the first element since it is all 0s and these is not a real pattern, more like the not-pattern
	input_patterns = create_input_patterns(number_of_patterns, number_of_patterns+4)

	#temp = input_patterns[5]
	input_patterns[2] = input_patterns[6]
	#input_patterns[5] = temp

	input_patterns = input_patterns[1:-3]

	print(input_patterns)

	data = []

	# first add recurring patterns at the beginning, then correct amount of random data
	for i in range(number_of_patterns):
		data.append(add_noise(input_patterns[i]))
	if(percentage_of_random_data > 0):
		for i in range(round(number_of_patterns*percentage_of_random_data/(1-percentage_of_random_data))):
			print("bla")
			push_random(data)

	while(len(data) < 1000):
		push_datum(data, input_patterns)		

	return np.array(data)

