from __future__ import division
import numpy as np
import math as m


class Kliep(object):

	def __init__(self, kliepParEta, kliepParLambda, kliepParB, kliepParThreshold, kliepDefSigma=0.01):
		self.kliepDefSigma = kliepDefSigma
		self.kliepParEta = kliepParEta
		self.kliepParLambda = kliepParLambda
		self.kliepParB = kliepParB
		self.kliepParThreshold = kliepParThreshold

	def pdf_gaussian(self, x, mu):
		x_size = np.shape(x)
		d = x_size[0]
		nx = x_size[len(x_size) - 1]

		tmp = (x - np.tile(mu, (1, nx)))/(np.sqrt(2) * np.tile(self.kliepDefSigma, (1, nx)))
		denom = m.pow((2 * m.pi), (-1 / 2))
		px = np.exp(-np.power(tmp, 2, dtype='float64'))*(denom / self.kliepDefSigma)
		return px

	def kernel_Gaussian(self, x, c):
		x_size = np.shape(x)
		dx = x_size[0]
		nx = x_size[len(x_size) - 1]

		c_size = np.shape(c)
		dc = c_size[0]
		nc = c_size[len(c_size) - 1]

		x2 = np.power(x, 2, dtype=np.float)
		c2 = np.power(c, 2, dtype=np.float)
		# if the array is 1D, need to add an axis first before doing transpose.
		distance2 = np.tile(c2, (nx, 1)) + np.tile(x2.T, (1, nc)) - (2 * x.T * c)
		X = np.exp(-distance2 / (2 * m.pow(self.kliepDefSigma, 2)), dtype='float64')
		return X

	"""
	- find kernel gaussians for multi dim x
	- in x and c, each column represents one data point
	"""

	def kernel_Gaussian_mdim(self, x, c):
		x_size = np.shape(x)
		dx = x_size[0]
		nx = x_size[len(x_size) - 1]

		c_size = np.shape(c)
		dc = c_size[0]
		nc = c_size[len(c_size) - 1]
		distance2 = None

		for i in range(0, nx):
			# though we extract a column, it become a row matrix in python.
			x_col_i = x[:, i]
			dist_x_col_i_c = self.distance(x_col_i[np.newaxis].T, dx, 1, c, dc, nc)
			if distance2 is None:
				# since X will have more rows, so while copying need to add a new axis also.
				distance2 = dist_x_col_i_c[np.newaxis]
			else:
				distance2 = np.append(distance2, dist_x_col_i_c[np.newaxis], axis=0)
		X = np.exp(-distance2 / (2 * m.pow(self.kliepDefSigma, 2)), dtype='float64')
		return X

	def kernel_Gaussian_mdim_choose_sigma(self, x, c, sigma):
		x_size = np.shape(x)
		dx = x_size[0]
		nx = x_size[len(x_size) - 1]

		c_size = np.shape(c)
		dc = c_size[0]
		nc = c_size[len(c_size) - 1]
		distance2 = None

		for i in range(0, nx):
			# though we extract a column, it become a row matrix in python.
			x_col_i = x[:, i]
			dist_x_col_i_c = self.distance(x_col_i[np.newaxis].T, dx, 1, c, dc, nc)
			if distance2 is None:
				# since X will have more rows, so while copying need to add a new axis also.
				distance2 = dist_x_col_i_c[np.newaxis]
			else:
				distance2 = np.append(distance2, dist_x_col_i_c[np.newaxis], axis=0)
		X = np.exp(-distance2/(2*m.pow(sigma, 2)), dtype='float64')
		return X

	"""
	x_col_i represents ith instance in row matrix format.
	c represents the selected test points. ith column represents ith selected test point.

	distance returns a row matrix, dimension 1*c_ncol, where (1,i) element is the distance between the instance
	represented by x_col_i and ith instance in c, i.e., ith column in c
	"""

	def distance(self, x_col_i, x_col_i_nrow, x_col_i_ncol, c, c_nrow, c_ncol):
		dist_tmp = np.power(np.tile(x_col_i, (1, c_ncol)) - c, 2, dtype='float64')
		# need to do column-wise sum
		dist_2 = np.sum(dist_tmp, axis=0, dtype='float64')
		return dist_2

	def KLIEP_projection(self, alpha, Xte, meanDistSrcData, c):
		# b_alpha = np.sum(b*alpha)
		b_alpha = np.dot(meanDistSrcData.T, alpha)
		alpha = alpha + meanDistSrcData * (1 - b_alpha) * np.linalg.pinv(c, rcond=1e-20)
		# alpha = np.max(0,alpha[np.newaxis])
		alpha[alpha < 0] = 0
		b_alpha_new = np.dot(meanDistSrcData.T, alpha)
		alpha = alpha * np.linalg.pinv(b_alpha_new, rcond=1e-20)
		Xte_alpha = np.dot(Xte, alpha)
		Xte_alpha[(Xte_alpha-0)<0.00001] = 0.00001
		#Xte_alpha_no_zeros = np.array([(1/100) if (h - 0) < 0.00001 else h for h in Xte_alpha])
		log_xte_alpha = np.log(Xte_alpha, dtype='float64')
		score = np.mean(log_xte_alpha, dtype='float64')
		return alpha, Xte_alpha, score

	def KLIEP_projection_wo_score(self, alpha, meanDistSrcData, c):
		b_alpha = np.dot(meanDistSrcData.T, alpha)
		alpha = alpha + meanDistSrcData * (1 - b_alpha) * np.linalg.pinv(c, rcond=1e-20)
		# alpha = np.max(0,alpha[np.newaxis])
		alpha[alpha < 0] = 0
		b_alpha_new = np.dot(meanDistSrcData.T, alpha)
		alpha = alpha * np.linalg.pinv(b_alpha_new, rcond=1e-20)
		return alpha

	def KLIEP_learning(self, mean_X_de, X_nu):
		X_nu_size = np.shape(X_nu)
		n_nu = X_nu_size[0]
		nc = X_nu_size[len(X_nu_size) - 1]

		max_iteration = 100
		epsilon_list = np.power(10, range(3, -4, -1), dtype='float64')
		# c = sum(np.power(mean_X_de, 2, dtype=np.float))
		c = np.dot(mean_X_de.T, mean_X_de)
		alpha = np.ones((nc, 1))

		[alpha, X_nu_alpha, score] = self.KLIEP_projection(alpha, X_nu, mean_X_de, c)

		for epsilon in epsilon_list:
			for iteration in range(1, max_iteration):
				alpha_tmp = alpha + (epsilon * np.dot(X_nu.T, (1 / X_nu_alpha)))
				[alpha_new, X_nu_alpha_new, score_new] = self.KLIEP_projection(alpha_tmp, X_nu, mean_X_de, c)
				if (score_new - score) <= 0:
					break
				score = score_new
				alpha = alpha_new
				X_nu_alpha = X_nu_alpha_new
		return alpha

	def KLIEP(self, srcData, trgData):
		srcDataSize = np.shape(srcData)
		nRowSrcData = srcDataSize[0]
		nColSrcData = srcDataSize[len(srcDataSize) - 1]

		trgDataSize = np.shape(trgData)
		nRowTrgData = trgDataSize[0]
		nColTrgData = trgDataSize[len(trgDataSize) - 1]

		b = min(self.kliepParB, nColTrgData)

		#rand_index = np.random.permutation(nColTrgData)
		# rand_index = genfromtxt('rand_index.csv', delimiter=',')-1
		#refPoints = trgData[:, rand_index[0:b].tolist()]
		refPoints = trgData[:, -b:]

		######### Computing the final solution wh_x_de
		kernelMatSrcData = self.kernel_Gaussian_mdim(srcData, refPoints)
		kernelMatTrgData = self.kernel_Gaussian_mdim(trgData, refPoints)
		meanDistSrcData = np.transpose(np.mean(kernelMatSrcData, 0)[np.newaxis])
		alphah = self.KLIEP_learning(meanDistSrcData, kernelMatTrgData)
		# wh_x_nu = np.transpose(np.dot(X_nu, alphah))
		#weightTrgData = np.dot(kernelMatTrgData, alphah)

		return alphah, kernelMatSrcData, kernelMatTrgData, refPoints


	def chooseSigma(self, srcData, trgData, fold=5):
		srcDataSize = np.shape(srcData)
		nRowSrcData = srcDataSize[0]
		nColSrcData = srcDataSize[len(srcDataSize) - 1]

		trgDataSize = np.shape(trgData)
		nRowTrgData = trgDataSize[0]
		nColTrgData = trgDataSize[len(trgDataSize) - 1]

		print "Choose Sigma"
		####### Choosing Gaussian kernel center `x_ce'
		# rand_index = np.random.permutation(n_nu)
		b = min(self.kliepParB, nColTrgData)

		# undo after finishing debug
		# x_ce = np.array(x_nu)
		# np.random.shuffle(x_ce)
		rand_index = np.random.permutation(nColTrgData)
		# rand_index = genfromtxt('rand_index.csv', delimiter=',')-1
		refPoints = trgData[:, rand_index[0:b].tolist()]

		####### Searching Gaussian kernel width `sigma_chosen'
		sigma = 10
		score = -float("inf")
		epsilon_list = range(int(m.log10(sigma)) - 1, -2, -1)
		for epsilon in epsilon_list:
			for iteration in range(1, 10, 1):
				sigma_new = sigma - m.pow(10, epsilon)
				print "sigma = ", sigma, " epsilon=", epsilon, "sigma_new=", sigma_new
				# undo after finishing debug
				cv_index = np.random.permutation(nColTrgData)
				# cv_index = genfromtxt('cv_index' + str(epsilon) + '_' + str(iteration) + '.csv', delimiter=',')-1

				cv_split = np.floor(np.divide(np.multiply(range(0, nColTrgData), fold), nColTrgData)) + 1
				score_new = 0

				kernelMatSrcData = self.kernel_Gaussian_mdim_choose_sigma(srcData, refPoints, sigma_new)
				kernelMatTrgData = self.kernel_Gaussian_mdim_choose_sigma(trgData, refPoints, sigma_new)
				# axis = 0 means column-wise mean
				meanDistSrcData = np.transpose(np.mean(kernelMatSrcData, axis=0)[np.newaxis])
				for i in range(1, fold + 1, 1):
					alpha_cv = self.KLIEP_learning(meanDistSrcData, kernelMatTrgData[cv_index[cv_split != i].tolist(), :])
					wh_cv = np.dot(kernelMatTrgData[cv_index[cv_split == i].tolist(), :], alpha_cv)
					score_new = score_new + (np.mean(np.log(wh_cv), dtype=np.float)/fold)

				if (score_new - score) <= 0:
					break
				score = score_new
				sigma = sigma_new
				print "score=", score, " sigma=", sigma, "epsilon=", epsilon, "iteration=", iteration

		print "Sigma = ", str(sigma)
		return sigma

	def changeDetection(self, trgData, refPointsOld, alphahOld, refPointsNew, alphahNew, kernelMatTrgDataNew=None):
		if len(np.shape(trgData)) == 1:
			trgData = trgData[np.newaxis]

		if kernelMatTrgDataNew is None:
			kernelMatTrgDataNew = self.kernel_Gaussian_mdim(trgData, refPointsNew)
		kernelMatTrgDataOld = self.kernel_Gaussian_mdim(trgData, refPointsOld)

		weightTrgDataNew = self.calcInstanceWeights(kernelMatTrgDataNew, alphahNew)
		weightTrgDataNew[(weightTrgDataNew - 0) < 0.00001] = 0.00001
		#weightTrgDataNew_no_zeros = np.array([float(0.0001) if (h-0)<0.00001 else h for h in weightTrgDataNew[0]])

		weightTrgDataOld = self.calcInstanceWeights(kernelMatTrgDataOld, alphahOld)
		weightTrgDataOld[(weightTrgDataOld - 0) < 0.00001] = 0.00001
		#weightTrgDataOld_no_zeros = np.array([float(0.0001) if (h - 0) < 0.00001 else h for h in weightTrgDataOld[0]])

		l_ratios = weightTrgDataNew/weightTrgDataOld

		lnWeightTrgData = np.log(l_ratios, dtype='float64')
		changeScore = np.sum(lnWeightTrgData, dtype='float64')
		#print "ChangeScore=", changeScore
		return changeScore > self.kliepParThreshold, changeScore, kernelMatTrgDataNew


	"""
	updateAlpha parameters:
	srcData - contains instances from src stream
	trgData - contains instances from trg stream, including the new point
	newTrgPoint - is the new point, last column of trgData should match with newTrgPoint
	alphah - most recent set of alpha
	"""
	def updateAlpha(self, srcData, trgData, newTrgPoint, refPoints, alphah, kernelMatSrcData=None):
		if len(np.shape(srcData)) == 1:
			srcData = srcData[np.newaxis]
		if len(np.shape(trgData)) == 1:
			trgData = trgData[np.newaxis]

		# calculate c
		trgDataSize = np.shape(trgData)
		nRowTrgData = trgDataSize[0]
		nColTrgData = trgDataSize[len(trgDataSize) - 1]

		if newTrgPoint.ndim == 1:
			newTrgPoint = newTrgPoint[np.newaxis]

		kernelNewTrgPoint = self.kernel_Gaussian_mdim(newTrgPoint, refPoints)
		# alphah is a column vector, each row of kernel_x_new represents distances for one data point
		c = np.dot(kernelNewTrgPoint, alphah)

		# update alpha values
		tmp = 1 - (self.kliepParEta * self.kliepParLambda)
		alphah = alphah * tmp
		alphah = alphah[1:, :]
		alphah = np.append(alphah, self.kliepParEta/c, axis=0)
		alphah, kernelMatSrcData = self.satConstraints(srcData, trgData, refPoints, alphah, kernelMatSrcData)

		return alphah, kernelMatSrcData


	def satConstraints(self, srcData, trgData, refPoints, alphah, kernelMatSrcData=None):
		trgDataSize = np.shape(trgData)
		nRowTrgData = trgDataSize[0]
		nColTrgData = trgDataSize[len(trgDataSize) - 1]

		if kernelMatSrcData is None:
			kernelMatSrcData = self.kernel_Gaussian_mdim(srcData, refPoints)
		meanDistSrcData = self.colWiseMeanTransposed(kernelMatSrcData)
		# c = sum(np.power(mean_X_de, 2, dtype=np.float))
		c = np.dot(meanDistSrcData.T, meanDistSrcData)
		alphah = self.KLIEP_projection_wo_score(alphah, meanDistSrcData, c)

		return alphah, kernelMatSrcData


	"""
	returns transpose of matrix resulting from taking column wise mean of mat.
	"""
	def colWiseMeanTransposed(self, mat):
		return np.transpose(np.mean(mat, 0)[np.newaxis])


	"""
	Returns instance weights as a row vector
	"""
	def calcInstanceWeights(self, kernelMat, alphah):
		return np.dot(kernelMat, alphah).T