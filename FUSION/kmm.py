import math, numpy, sklearn.metrics.pairwise as sk, sys
from sklearn import linear_model
from cvxopt import matrix, solvers

#DENSITY ESTIMATION
#KMM solving the quadratic programming problem to get betas (weights) for each training instance
def kmm(Xtrain, Xtest, sigma):
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


#KMM PARAMETER TUNING
#Train a linear regression model with Lasso (L1 regularization).
#Model parameter selection via cross validation
#Predict the target (Beta) for a given test dataset
def regression(XTrain, betaTrain, XTest):
	model = linear_model.LassoCV(cv=10, alphas=[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10])
	model.fit(XTrain, betaTrain)
	Beta = model.predict(XTest)
	return [i for i in Beta]


#KMM PARAMETER TUNING
#Compute J score for parameter tuning of KMM
def computeJ(betaTrain, betaTest):
	tr = sum([i ** 2 for i in betaTrain])
	te = sum(betaTest)
	return ((1/float(len(betaTrain)))*tr) - ((2/float(len(betaTest)))*te)


#I/O OPERATIONS
#Read input csv file
def getData(filename):
	data = []
	with open(filename) as f:
		content = f.readlines()
	
	for line in content:
		line = line.strip()
		data.append(map(float,line.split(",")))
	return data


#I/O OPERATIONS
#Write Output to file
def writeFile(filename, data):
	if len(data) == 0:
		return
	
	with open(filename, 'w') as f:
		for i in data:
			f.write(str(i) + '\n')


#MAIN ALGORITHM
#compute beta
def getBeta(traindata, testdata, gammab):
	
	Jmin = 0
	beta = []
	
	for g in gammab:
		betaTrain = kmm(traindata, testdata, g)
		betaTest = regression(traindata, betaTrain, testdata)
		J = computeJ(betaTrain,betaTest)
		
		#print betaTrain
		#print betaTest
		#print J
		
		if len(beta) == 0:
			Jmin = J
			beta = list(betaTrain)
		elif Jmin > J:
			Jmin = J
			beta = list(betaTrain)
	
	return beta
    

#MAIN METHOD
def main():
    #traindata = [[1,2,3],[4,7,4],[3,3,3],[4,4,4],[5,5,5],[3,4,5],[1,2,3],[4,7,4],[3,3,3],[4,4,4],[5,5,5],[3,4,5],[1,2,3],[4,7,4],[3,3,3],[4,4,4],[5,5,5],[3,4,5],[1,2,3],[4,7,4],[3,3,3],[4,4,4],[5,5,5],[3,4,5]]
    #testdata = [[5,9,10],[4,5,6],[10,20,30],[1,2,3],[3,4,5],[5,6,7],[7,8,9],[100,100,100],[11,22,33],[12,11,5],[5,9,10],[4,5,6],[10,20,30],[1,2,3],[3,4,5],[5,6,7],[7,8,9],[100,100,100],[11,22,33],[12,11,5]]
    #gammab = [0.001]
	
	if len(sys.argv) != 4:
		print 'Incorrect number of arguments.'
		print 'Arg: training_file, test_file, output_file.'
		return
    
	traindata = getData(sys.argv[1])
	testdata = getData(sys.argv[2])
	gammab = [1/float(len(traindata)),0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10]
	print 'Got training and test data.'
	
	beta = getBeta(traindata, testdata, gammab)
	
	writeFile(sys.argv[3], beta)

if __name__ == '__main__':
	main()
