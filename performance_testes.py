
import timeit
import numpy as np
from numpy import array, arange, vectorize, rint
import math



a = np.array([0, 1, 1])
b = np.array([0, 0, 1])

print(a*b)



input_mean = 7
def testapply(a):
	a = [ a[0] + math.copysign(0.1, a[0]) if abs(a[0]*a[1]) < abs(input_mean) else a[0] - math.copysign(0.1, a[0]), a[1] ]
	return a

def testapply2(a, b):
	a = a + math.copysign(0.1, a) if abs(a*b) < abs(input_mean) else a - math.copysign(0.1, a)
	return [a, b]


# SETUP
arr = np.array([1,2,3,4,5,6], dtype=float)
brr = np.array([6,5,4,3,2,1], dtype=float)
get_arrays = lambda : np.array([arr, brr])
#dim = lambda x : int(round(x * 0.67328))

# TIMER
def best(fname, reps, side):
    global a
    a = get_arrays()
    t = timeit.Timer('%s(a)' % fname,
                     setup='from __main__ import %s, a' % fname)
    return min(t.repeat(reps, 3))  #low num as in place --> converge to 1

# FUNCTIONS

def apply_axis(arrays_):
	arrays_ = arrays_.T
	arrays_ = np.apply_along_axis(testapply, 1, arrays_)
	return arrays_

def mappenyes(arrays_):
	arrays_ = arrays_.T
	
	return np.array(list(map(testapply, arrays_)))
	

#return np.fromiter(map(testapply, arrays_)

# ¯\_(ツ)_/¯
def vectorize(array_):
	array_ = array_.T


	def vectorize2(funcs):
		def fnv(arr):
			return np.vstack([f(arr) for f in funcs])
		return fnv

	f2 = vectorize2((lambda x : testapply(x), lambda x : testapply(x), lambda x : testapply(x), lambda x : testapply(x), lambda x : testapply(x), lambda x : testapply(x)))

	#print(array_)
	array_ = f2(array_)
	#print(array_)

	#array_ = array_.T
	
	#fn = np.vectorize(testapply2)
	#return fn(np.array(array_[0]), np.array(array_[1]))
	return array_

def fromfunc(arrays_):
	arrays_ = np.array(arrays_).T
	#return np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
	#return np.fromfunction(testapply2, arrays_, dtype=int)

def forloop(arrays_):
	#arrays_ = np.array(arrays_, dtype=float).T
	arrays_ = arrays_.T

	for idx in range(len(arrays_)):
		arrays_[idx] = testapply(arrays_[idx])

	return arrays_
	

fns = [apply_axis, mappenyes, forloop, fromfunc]
fnames = ['apply_axis', 'mappenyes', 'forloop', 'fromfunc']

# MAIN
r = []
for idx, fname in enumerate(fnames):
    print('\nTesting `%s`...' % fname)
    r.append(best(fname, reps=50, side=50))
    # The following is for visually checking the functions returns same results
    #global tm
    tmp = get_arrays()    
    print(tmp)
    tmp = fns[idx](tmp)
    #eval('tmp = %s(tmp)' % fname)
    print (tmp)


tmp = min(r)/100
print('\n===== ...AND THE WINNER IS... =========================')
for idx, fname in enumerate(fnames):
	print(fname, "\t", r[idx]*1000)
print('=======================================================\n')
