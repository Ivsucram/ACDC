import math, sklearn.metrics.pairwise as sk
from sklearn import svm
import numpy as np
import random, sys


class Model(object):

	def __init__(self):
		self.model = None
		self.weight = 0.0

	"""
	Initialize training of a new weighted SVM model by choosing best parameters.
	Sets the trained model for this object.
	"""
	def trainUsingKLIEPWeights(self, traindata, trainLabels, weightSrcData, maxvar, svmC, svmGamma, svmKernel):
		self.model = svm.SVC(decision_function_shape='ovr', probability=True, C=svmC, gamma=svmGamma, kernel=svmKernel)
		self.model.fit(traindata, trainLabels)

	"""
	Test the weighted SVM to predict labels of a given test data.
	Returns the result of prediction, and confidence behind the prediction
	"""
	def test(self, testdata):
		#predict and gather results
		#predictedClass = ["" for x in range(len(testdata))]
		#confidences = np.zeros(len(testdata))
		confidences = []
		#reshapedData = np.reshape(testdata, (1,-1))
		if len(testdata)==1:
			testdata = np.reshape(testdata, (1,-1))
		predictions = self.model.predict(testdata)
		probs = self.model.predict_proba(testdata)
		for i in range(0, len(testdata)):
			#curData = np.reshape(testdata[i], (1,-1))
			#predictedClass[i] = self.model.predict(curData)[0]
			for j in range(len(self.model.classes_)):
				if self.model.classes_[j] == predictions[i]:
					#confidences[i] = prob[j]
					confidences.append(probs[i][j])
					break
			"""
			scores = self.model.decision_function(curData)

			if len(self.model.classes_)<=2:
				confidences[i] = min(1.0, math.fabs(scores[0]))
			else:
				# we calculate the confidence by taking normalized score
				totScore = 0.0
				for x, y in zip(self.model.classes_, scores[0]):
					totScore += math.fabs(y)
					if predictedClass[i] == x:
						confidences[i] = math.fabs(y)
				confidences[i] /= totScore
			"""
		return predictions, confidences

	"""
	Set model weights using test prediction.
	For source weight, use error rate with known source data labels.
	For target weight, use confidence (or probability) measure on target data.
	"""
	def computeModelWeightKLIEP(self, data, maxvar):
		totConf = 0.0
		predictedClass, confidences = self.test(data)
		for i in range(0, len(confidences)):
			totConf += confidences[i]
		return totConf/len(data)