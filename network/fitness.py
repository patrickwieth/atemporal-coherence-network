import numpy as np

def norm(data):
	maxval = np.amax(data)
	if(maxval == 0):	return data 	
	else:				return data / maxval

def discrimination(matrix):

	matrix = norm(matrix)

	def discriminate(line):
		maxi = np.argmax(line)

		sum = 0
		for i, val in enumerate(line):
			if(i == maxi): 	sum += val
			else: 			sum -= val

		return sum

	row_sums = list(map(discriminate, matrix))
	col_sums = list(map(discriminate, np.transpose(matrix)))

	return (np.sum(row_sums) + np.sum(col_sums)) / (len(row_sums) + len(col_sums))