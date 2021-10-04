import math
from model import Model
from properties import Properties


class Ensemble(object):

	def __init__(self, ensemble_size):
		self.models = []
		self.size = ensemble_size


	"""
	Update weights for all models in the ensemble.
	"""
	def updateWeight(self, data, isSource):
		for m in self.models:
			m.computeModelWeight(data, isSource, Properties.MAXVAR)

	def reEvalModelWeights(self, data, maxvar):
		for i in range(0, len(self.models)):
			self.models[i].weight = self.models[i].computeModelWeightKLIEP(data, maxvar)


	"""
	Adding a new model to the Ensemble.
	Returns the index of the Ensemble array where the model is added.
	"""
	def __addModelKLIEP(self, model, data, maxvar):
		index = 0
		self.reEvalModelWeights(data, maxvar)
		if len(self.models) < self.size:
			self.models.append(model)
			index = len(self.models)-1
		else:
			#replace least desirable model
			index = self.__getLeastDesirableModelKLIEP()
			if self.models[index].weight < model.weight:
				Properties.logger.info('Least desirable model removed at ' + str(index))
				self.models[index] = model
			else:
				Properties.logger.info('New model was not added as its weight is less than all of the existing models')
				return -1
		return index


	"""
	Adding a new model to the Ensemble.
	Returns the index of the Ensemble array where the model is added.
	"""
	def __addModel(self, model):
		index = 0
		if len(self.models) < self.size:
			self.models.append(model)
			index = len(self.models)-1
		else:
			#replace least desirable model
			index = self.__getLeastDesirableModel()
			Properties.logger.info('Least desirable model removed at ' + str(index))
			self.models[index] = model
		return index


	"""
	Compute the least desirable model to be replaced when the ensemble size has reached its limit.
	Least desirable is one having least target weight
	Returns the array index of the least desired model.
	"""
	def __getLeastDesirableModelKLIEP(self):
		weights = {}
		for i in xrange(len(self.models)):
			weights[i] = self.models[i].weight

		keys = sorted(weights, key=weights.get)

		return keys[0]


	"""
	Compute the least desirable model to be replaced when the ensemble size has reached its limit.
	Least desirable is one having least target weight, but not the largest source weight.
	Returns the array index of the least desired model.
	"""
	def __getLeastDesirableModel(self):
		sweights = {}
		tweights = {}
		for i in xrange(len(self.models)):
			sweights[i] = self.models[i].sweight
			tweights[i] = self.models[i].tweight

		skeys = sorted(sweights, reverse=True, key=sweights.get)
		tkeys = sorted(tweights, key=tweights.get)

		# skeys = sweights.keys()
		# tkeys = tweights.keys()

		for i in xrange(len(skeys)):
			if tkeys[i] == skeys[i]:
				continue
			else:
				return tkeys[i]

		return tkeys[0]


	"""
	Initiate the creation of appropriate model in the ensemble for given target data.
	Also compute weights for the new model based on the current data.
	"""
	def generateNewModelKLIEP(self, srcData, srcLabels, trgData, weightSrcData, svmC, svmGamma, svmKernel):
		model = Model()

		if len(srcData) == 0 or len(trgData) == 0:
			raise Exception('Source or Target stream should have some elements')

		#Create new model
		Properties.logger.info('Target model creation')
		model.trainUsingKLIEPWeights(srcData, srcLabels, weightSrcData, Properties.MAXVAR, svmC, svmGamma, svmKernel)

		#compute source and target weight
		Properties.logger.info('Computing model weights')
		model.weight = model.computeModelWeightKLIEP(trgData, Properties.MAXVAR)

		#update ensemble
		index = self.__addModelKLIEP(model, trgData, Properties.MAXVAR)
		if index != -1:
			Properties.logger.info('Ensemble updated at ' + str(index))


	"""
	Initiate the creation of appropriate model in the ensemble for given source or target data.
	Also compute weights for the new model based on the current data.
	"""
	def generateNewModel(self, sourceData, targetData, isSource, useSvmCVParams, svmDefC, svmDefGamma):
		model = Model()

		if len(sourceData) == 0 or len(targetData) == 0:
			raise Exception('Source or Target stream should have some elements')

		#Create new model
		if isSource:
			Properties.logger.info('Source model creation')
			model.train(sourceData, None, Properties.MAXVAR, useSvmCVParams, svmDefC, svmDefGamma)
		else:
			Properties.logger.info('Target model creation')
			model.train(sourceData, targetData, Properties.MAXVAR, useSvmCVParams, svmDefC, svmDefGamma)

		#compute source and target weight
		Properties.logger.info('Computing model weights')
		model.computeModelWeight(sourceData, True, Properties.MAXVAR)
		model.computeModelWeight(targetData, False, Properties.MAXVAR)

		#update ensemble
		index = self.__addModel(model)
		Properties.logger.info('Ensemble updated at ' + str(index))


	"""
	Get prediction for a given data instance from each model.
	For source data: Ensemble prediction is 1 if maximum weighted vote class label matches true class label, else 0.
	For target data: Ensemble prediction class with max weighted vote class label, and average (for all class) confidence measure.
	"""

	def evaluateEnsembleKLIEP(self, dataInstance):
		confSum = {}
		weightSum = {}
		for m in self.models:
			# test data instance in each model
			predictedClass, confidence = m.test(dataInstance)
			# gather result
			if predictedClass[0] in confSum:
				confSum[predictedClass[0]] += confidence[0]
				weightSum[predictedClass[0]] += m.weight
			else:
				confSum[predictedClass[0]] = confidence[0]
				weightSum[predictedClass[0]] = m.weight

		# get maximum voted class label
		classMax = 0.0
		sorted(confSum, key=confSum.get, reverse=True)
		classMax = confSum.keys()[0]

		return [classMax, confSum[classMax]/len(self.models)]

	"""
	Get prediction for a given data instance from each model.
	For source data: Ensemble prediction is 1 if maximum weighted vote class label matches true class label, else 0.
	For target data: Ensemble prediction class with max weighted vote class label, and average (for all class) confidence measure.
	"""
	def evaluateEnsemble(self, dataInstance, isSource):

		classSum = {}
		for m in self.models:
			#test data instance in each model
			result = m.test([dataInstance], Properties.MAXVAR)
			#gather result
			if isSource:
				if int(result[0][0]) in classSum:
					classSum[int(result[0][0])] += m.sweight
				else:
					classSum[int(result[0][0])] = m.sweight
			else:
				if int(result[0][0]) in classSum:
					classSum[int(result[0][0])] += result[0][1]
				else:
					classSum[int(result[0][0])] = result[0][1]

		#get maximum voted sum class label
		classMax = 0.0
		sumMax = max(classSum.values())
		for i in classSum:
			if classSum[i] == sumMax:
				classMax = i

		if isSource:
			#for source data, check true vs predicted class label
			if classMax == dataInstance[-1]:
				return [1, -1]
			else:
				return [0, -1]
		else:
			# for target data
			return [classMax, sumMax/len(self.models)]

	"""
	Get summary of models in ensemble.
	"""
	def getEnsembleSummary(self):
		summry = '************************* E N S E M B L E    S U M M A R Y ************************\n'
		summry += 'Ensemble has currently ' + str(len(self.models)) + ' models.\n'
		for i in xrange(len(self.models)):
			summry += 'Model' + str(i+1) + ': weights<' + str(self.models[i].weight) + '>\n'
		return summry



			


