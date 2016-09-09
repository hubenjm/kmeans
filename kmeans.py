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

def compute_objective(X, Mu, nearest_mean):
	result = 0.0
	for j in xrange(Mu.shape[1]):
		result += np.linalg.norm(X[:, nearest_mean == j] - Mu[:,j].reshape((Mu.shape[0], 1)))**2
	
	return result

def update_mu(X, Mu, nearest_mean):
	maxdiff = 0.0
	for j in range(Mu.shape[1]):
		muj_old = Mu[:,j].copy()
		Mu[:,j] = np.sum(X[:, nearest_mean == j], axis = 1)/np.sum(nearest_mean == j)
		maxdiff = max(maxdiff, np.linalg.norm(muj_old - Mu[:,j]))

	return maxdiff

def find_k_means(X, k, seed = 1):
	np.random.seed(seed)
	n = X.shape[0] #dimension of space
	m = X.shape[1] #number of points

	assert X.shape[1] >= k
	index = np.random.choice(m, size = k, replace = False)

	Mu = X[:, index] #initial means, selected randomly from columns of X

	#initialize closest mean	
	nearest_mean = np.zeros(m, dtype = np.int)
	mindist = np.zeros(m)
	for j in xrange(m):
		nearest_mean[j], mindist[j] = find_closest_mean(X[:,j], Mu)
	
	while update_mu(X, Mu, nearest_mean) > 0:
		for j in xrange(m):
			nearest_mean[j], mindist[j] = find_closest_mean(X[:,j], Mu)
		print "*",
		sys.stdout.flush()

	return Mu, nearest_mean, mindist


def test():
	np.random.seed(1)
	X = np.random.normal(size = (2, 10000))
	k = 6
	Mu, nearest_mean, mindist = find_k_means(X, k)
	colors = ["r", "b", "g"]

	for j in range(k):
		plt.plot(X[0,nearest_mean==j], X[1,nearest_mean==j], colors[j%3]+'.')
		
	plt.plot(Mu[0,:], Mu[1,:], 'k.', markersize=20)
	plt.show()
	

	#print Mu, nearest_mean, mindist

if __name__ == "__main__":
	test()
	

	
	
	
