import math, numpy, sklearn.metrics.pairwise as sk
from cvxopt import matrix, solvers
from svmutil import *
from grid import *
import random, sys


class Model(object):

	def __init__(self):
		self.model = None
		self.sweight = 1.0
		self.tweight = 1.0

		self.__trainLabelOrder = []


	"""
	Compute instance (importance) weights using Kernel Mean Matching.
	Returns a list of instance weights for training data.
	"""
	def __kmm(self, Xtrain, Xtest, sigma):
		n_tr = len(Xtrain)
		n_te = len(Xtest)

		#calculate Kernel
		print 'Computing kernel for training data ...'
		K_ns = sk.rbf_kernel(Xtrain, Xtrain, sigma)
		#make it symmetric
		K = 0.5*(K_ns + K_ns.transpose())

		#calculate kappa
		print 'Computing kernel for kappa ...'
		kappa_r = sk.rbf_kernel(Xtrain, Xtest, sigma)
		ones = numpy.ones(shape=(n_te, 1))
		kappa = numpy.dot(kappa_r, ones)
		kappa = -(float(n_tr)/float(n_te)) * kappa

		#calculate eps
		eps = (math.sqrt(n_tr) - 1)/math.sqrt(n_tr)

		#constraints
		A0 = numpy.ones(shape=(1,n_tr))
		A1 = -numpy.ones(shape=(1,n_tr))
		A = numpy.vstack([A0, A1, -numpy.eye(n_tr), numpy.eye(n_tr)])
		b = numpy.array([[n_tr*(eps+1), n_tr*(eps-1)]])
		b = numpy.vstack([b.T, -numpy.zeros(shape=(n_tr,1)), numpy.ones(shape=(n_tr,1))*1000])

		print 'Solving quadratic program for beta ...'
		P = matrix(K, tc='d')
		q = matrix(kappa, tc='d')
		G = matrix(A, tc='d')
		h = matrix(b, tc='d')
		beta = solvers.qp(P,q,G,h)
		return [i for i in beta['x']]


	"""
	Build a SVM model.
	"""
	def __build(self, trainX, trainY, beta, svmParam):
		prob = svm_problem(beta, trainY, trainX)
		# param = svm_parameter('-s 0 -c 131072 -t 2 -q -b 1 -g 0.0001')
		param = svm_parameter('-s 0 -t 2 -q -b 1 -c ' + str(svmParam['c']) + ' -g ' + str(svmParam['g']))
		return svm_train(prob, param)


	# """
	# Compute distance between two
	# """
	# def __computeDistanceSq(self, d1, d2):
	# 	dist = 0
	# 	for i in d1:
	# 		if i in d2:
	# 			#when d1 and d2 have the same feature
	# 			dist += ((d1[i] - d2[i]) ** 2)
	# 		else:
	# 			#feature in d1 only
	# 			dist += (d1[i] ** 2)
	# 	for i in d2:
	# 		#feature in d2 only
	# 		if i not in d1:
	# 			dist += (d2[i] ** 2)
	# 	return dist



	"""
	Kernel width is the median of distances between instances of sparse data
	"""
	def __computeKernelWidth(self, data):
		dist = []
		for i in xrange(len(data)):
			for j in range(i+1, len(data)):
				# s = self.__computeDistanceSq(data[i], data[j])
				# dist.append(math.sqrt(s))
				dist.append(numpy.sqrt(numpy.sum((numpy.array(data[i]) - numpy.array(data[j])) ** 2)))
		return numpy.median(numpy.array(dist))



	"""
	Initialize training of a new weighted SVM model by choosing best parameters.
	Sets the trained model for this object.
	"""
	def train(self, traindata, testdata, maxvar):
		beta = []
		trainY = []
		trainX = []
		testX = []

		#SVM parameter selection
		# with open('train_svmpar.data', 'w') as f:
		# 	for d in traindata:
		# 		# if d[-1] not in self.__trainLabelOrder:
		# 		# 	self.__trainLabelOrder.append(d[-1])
		# 		line = str(d[-1])
		# 		for c in sorted(d):
		# 			if c != -1:
		# 				line += ' ' + str(c) + ':' + str(d[c])
		# 		f.write(line + '\n')
		# rate, svmParam = find_parameters('train_svmpar.data', '-log2c 1,100,10 -log2g -10,0,2 -gnuplot null -out null')

		svmParam = {'c':131072, 'g':0.0001}

		#Subsample training data if given data size is more than 1000
		newtraindata = []
		if len(traindata) <= 1000:
			newtraindata = traindata
		else:
			seen = []
			for i in xrange(1000):
				r = random.randint(0, 1000)
				if r not in seen:
					seen.append(r)
					newtraindata.append(traindata[r])

		#Data preparation for computing beta.
		#Data format: space separated <index:value> with class index as -1.
		for d in newtraindata:
			if d[-1] not in self.__trainLabelOrder:
				self.__trainLabelOrder.append(d[-1])
			trainY.append(d[-1])

			covar = []
			for c in xrange(maxvar):
				if c in d:
					covar.append(d[c])
				else:
					covar.append(0.0)
			trainX.append(covar)


		if testdata == None:
			for c in xrange(len(trainX)):
				beta.append(1.0)
		else:
			# gammab = 0.001
			gammab = self.__computeKernelWidth(trainX)
			for d in testdata:
				covar = []
				for c in xrange(maxvar):
					if c in d:
						covar.append(d[c])
					else:
						covar.append(0.0)
				testX.append(covar)

			beta = self.__kmm(trainX, testX, gammab)

		#Model training
		self.model = self.__build(trainX, trainY, beta, svmParam)


	"""
	Test the weighted SVM to predict labels of a given test data.
	Returns the result of prediction, each of the form <label, probability, true label>
	"""
	def test(self, testdata, maxvar):
		#Data preparation for model prediction
		#Data format: space separated <index:value> with class index as -1.
		testX = []
		testY = []
		for d in testdata:
			# if d[-1] not in self.__trainLabelOrder:
			# 	self.__trainLabelOrder.append(d[-1])
			testY.append(d[-1])
			covar = []
			for c in xrange(maxvar):
				if c in d:
					covar.append(d[c])
				else:
					covar.append(0.0)
			testX.append(covar)

		#predict and gather results
		res = svm_predict(testY, testX, self.model, '-q -b 1') #returns <label, accuracy, value>
		result = []
		for i in xrange(len(res[0])):
			result.append([res[0][i], res[2][i][self.__trainLabelOrder.index(res[0][i])], testY[i]])
		return result


	"""
	Compute weight of a source model using its error rate
	"""
	def __computeWeight(self, errorRate):
		if errorRate <= 0.5:
			if errorRate == 0:
				errorRate = 0.01
			return 0.5*math.log((1-errorRate)/errorRate)
		else:
			return 0.01


	"""
	Set model weights using test prediction.
	For source weight, use error rate with known source data labels.
	For target weight, use confidence (or probability) measure on target data.
	"""
	def computeModelWeight(self, data, isSource, maxvar):
		result = self.test(data, maxvar)
		if isSource:
			#for source weight
			err = 0
			for i in xrange(len(result)):
				if result[i][0] != data[i][-1]:
					err += 1
			self.sweight = self.__computeWeight(float(err)/len(data))
		else:
			#for target weight
			conf = 0.0
			for r in result:
				conf += r[1]
			self.tweight = (conf/len(result))



"""
FOR TESTING
"""
if __name__ == '__main__':
	traindata = []
	testdata = []
	labels = []
	maxvar = 5
	for i in xrange(10):
		y = random.randint(0,2)
		x = {-1:y}
		for j in xrange(maxvar):
			x[j] = (random.randint(0,100))

		if y not in labels:
			labels.append(y)
		traindata.append(x)

	for i in xrange(5):
		y = random.randint(0,2)
		x = {-1:y}
		for j in xrange(maxvar):
			x[j] = (random.randint(0,100))

		testdata.append(x)

	model = Model()
	model.train(traindata,testdata, maxvar)
	model.test(testdata, maxvar)
	print labels
