import numpy as np

def read_parameters():
	return np.load('results/parameter.npy')

def dump_parameters(paras):
	np.save("results/parameter.npy", paras)
