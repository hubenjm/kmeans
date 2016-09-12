import numpy as np
import matplotlib.pyplot as plt
import sys

def find_closest_mean(x, Mu):
	"""
	x is a 1d numpy array indicating a single point
	Mu is a matrix whose columns are current mean vectors

	"""
	D = Mu - np.reshape(x, (len(x), 1))
	v = np.sum(D*D, axis = 0)
	j = np.argmin(v)
	return j, v[j]

def dfind_closest_mean(x, dx, Mu, dMu):
	
	D = Mu - np.reshape(x, (len(x), 1)) #shape = (n, k)
	v = np.sum(D*D, axis = 0) #shape = (k,)

	dv = 2.0*D*(dMu - np.reshape(dx, (len(dx), 1))) #shape (len(dx), k)
	j = np.argmin(v)
	return j, v[j], np.sum(dv[:,j])
	
#this function isn't used
def compute_objective(X, Mu, nearest_mean):
	result = 0.0
	for j in xrange(Mu.shape[1]):
		result += np.linalg.norm(X[:, nearest_mean == j] - Mu[:,j].reshape((Mu.shape[0], 1)))**2
	
	return result
##########################

def update_mu(X, Mu, nearest_mean):
	"""
	nearest_mean is a np.array of ints of length X.shape[1] (num points)
	each entry represents the index of Mu corresponding to the closest mean to the jth point of X
	"""
	max_diff = 0.0
	for j in range(Mu.shape[1]):
		muj_old = Mu[:,j].copy()
		Mu[:,j] = np.sum(X[:, nearest_mean == j], axis = 1)/np.sum(nearest_mean == j) #average all points of X closest to Mu[:,j]
		max_diff = max(max_diff, np.linalg.norm(muj_old - Mu[:,j]))

	return max_diff

def dupdate_mu(X, dX, Mu, dMu, nearest_mean):
	"""
	nearest_mean is a np.array of ints of length X.shape[1] (num points)
	each entry represents the index of Mu corresponding to the closest mean to the jth point of X
	"""
	max_diff = 0.0
	for j in range(Mu.shape[1]):
		muj_old = Mu[:,j].copy()
		Mu[:,j] = np.sum(X[:, nearest_mean == j], axis = 1)/np.sum(nearest_mean == j) #average all points of X closest to Mu[:,j]
		dMu[:,j] = np.sum(dX[:, nearest_mean == j], axis = 1)/np.sum(nearest_mean == j)
		max_diff = max(max_diff, np.linalg.norm(muj_old - Mu[:,j]))

	return max_diff

def find_k_means(X, k, seed = 1):
	np.random.seed(seed)
	n = X.shape[0] #dimension of space
	m = X.shape[1] #number of points

	assert X.shape[1] >= k
	index = np.random.choice(m, size = k, replace = False) #initial choice of means indices

	Mu = X[:, index] #initial means, selected randomly from columns of X

	#initialize closest mean	
	nearest_mean = np.zeros(m, dtype = np.int)
	min_dist = np.zeros(m)
	for j in xrange(m):
		nearest_mean[j], min_dist[j] = find_closest_mean(X[:,j], Mu)
	
	while update_mu(X, Mu, nearest_mean) > 0:
		for j in xrange(m):
			nearest_mean[j], min_dist[j] = find_closest_mean(X[:,j], Mu)
		print "*",
		sys.stdout.flush()

	print ""
	return Mu, nearest_mean, min_dist

def dfind_k_means(X, dX, k, seed = 1):
	np.random.seed(seed)
	n = X.shape[0] #dimension of space
	m = X.shape[1] #number of points

	assert X.shape[1] >= k
	index = np.random.choice(m, size = k, replace = False) #initial choice of means indices

	Mu = X[:, index] #initial means, selected randomly from columns of X
	dMu = np.zeros(Mu.shape)

	#initialize closest mean	
	nearest_mean = np.zeros(m, dtype = np.int)
	min_dist = np.zeros(m)
	dmin_dist = np.zeros(m)
	for j in xrange(m):
		nearest_mean[j], min_dist[j], dmin_dist[j] = dfind_closest_mean(X[:,j], dX[:,j], Mu, dMu)
	
	while dupdate_mu(X, dX, Mu, dMu, nearest_mean) > 0:
		for j in xrange(m):
			nearest_mean[j], min_dist[j], dmin_dist[j] = dfind_closest_mean(X[:,j], dX[:,j], Mu, dMu)
		print "*",
		sys.stdout.flush()

	print ""
	return Mu, dMu, nearest_mean, min_dist, dmin_dist


def test1():
	np.random.seed(1)
	X = np.random.normal(size = (2, 10000))
	k = 2
	Mu, nearest_mean, min_dist = find_k_means(X, k)
	colors = ["r", "b", "g"]

	for j in range(k):
		plt.plot(X[0,nearest_mean==j], X[1,nearest_mean==j], colors[j%3]+'.')
		
	plt.plot(Mu[0,:], Mu[1,:], 'k.', markersize=20)
	plt.show()
	

	#print Mu, nearest_mean, mindist

def test2():
	m = 1000
	n = 2
	k = 6

	np.random.seed(1)
	X = np.random.normal(size = (n, m))
#	dX = np.random.normal(size = (n, m))
	dX = np.zeros((n, m))
	dX[:,0] = np.ones(n)

	Mu, dMu, nearest_mean, min_dist, dmin_dist = dfind_k_means(X, dX, k)
	colors = ["r", "b", "g"]

	for j in range(k):
		plt.plot(X[0,nearest_mean==j], X[1,nearest_mean==j], colors[j%3]+'.')
		
	plt.plot(Mu[0,:], Mu[1,:], 'k.', markersize=20)
	plt.show()

	print dMu

	epsilon = 1e-8
	Mu1, nearest_mean1, min_dist1 = find_k_means(X + epsilon*dX, k)
	dMu_approx = (Mu1 - Mu)/epsilon
	print dMu_approx

if __name__ == "__main__":
	test2()
	

	
	
	
