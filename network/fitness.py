import numpy as np

def norm(data):
	maxval = np.amax(data)
	minval = np.amin(data)

	if(maxval < -1*minval): maxval = -1*minval

	if(maxval == 0):	return data 	
	else:				return data / maxval 

def discrimination(matrix):

	oldmatrix = matrix
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

	result = (np.sum(row_sums) + np.sum(col_sums)) / (len(row_sums) + len(col_sums))

	if(result > 1.0):
		print ("RESULT >1 bug! set fitness to -1")
		print (matrix)
		return -1

	return (np.sum(row_sums) + np.sum(col_sums)) / (len(row_sums) + len(col_sums))