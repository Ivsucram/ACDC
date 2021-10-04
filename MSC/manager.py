from __future__ import print_function
from properties import Properties
from changedetection import ChangeDetection
from ensemble import Ensemble
from stream import Stream
from model import Model
import time, sys
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters
import numpy as np


class Manager(object):

	def __init__(self, sourceFile, targetFile):
		self.SWindow = []
		self.TWindow = []
		self.TPredictWindow = []

		self.SDataBuffer = [] #Queue
		self.TDataBuffer = [] #Queue

		self.SInitialDataBuffer = []
		self.TInitialDataBuffer = []

		self.changeDetector = ChangeDetection(Properties.GAMMA, Properties.SENSITIVITY, Properties.MAX_WINDOW_SIZE)
		self.ensemble = Ensemble(Properties.ENSEMBLE_SIZE)

		classNameList = []
		self.source = Stream(sourceFile, classNameList, Properties.INITIAL_DATA_SIZE)
		self.target = Stream(targetFile, classNameList, Properties.INITIAL_DATA_SIZE)
		Properties.MAXVAR = self.source.MAXVAR

		self.gateway = JavaGateway(start_callback_server=True, gateway_parameters=GatewayParameters(port=Properties.PY4JPORT), callback_server_parameters=CallbackServerParameters(port=Properties.PY4JPORT+1))
		self.app = self.gateway.entry_point


	"""
	Detect drift on a given data stream.
	Returns the change point index on the stream array.
	"""
	def __detectDrift(self, slidingWindow, flagStream):
		changePoint = -1
		if flagStream == 0:
			changePoint = self.changeDetector.detectSourceChange(slidingWindow)
		elif flagStream == 1:
			changePoint = self.changeDetector.detectTargetChange(slidingWindow)
		else:
			raise Exception('flagStream var has value ' + str(flagStream) + ' that is not supported.')
		return changePoint


	def __detectDriftJava(self, slidingWindow, flagStream):
		changePoint = -1

		sw = self.gateway.jvm.java.util.ArrayList()
		for i in xrange(len(slidingWindow)):
			sw.append(float(slidingWindow[i]))

		if flagStream == 0:
			changePoint = self.app.detectSourceChange(sw)
		elif flagStream == 1:
			changePoint = self.app.detectTargetChange(sw)
		else:
			raise Exception('flagStream var has value ' + str(flagStream) + ' that is not supported.')
		# print('ChangePoint = ' + str(changePoint))

		return changePoint



	"""
	Write value (accuracy or confidence) to a file with DatasetName as an identifier.
	"""
	def __saveResult(self, acc, datasetName):
		with open(datasetName + '_' + Properties.OUTFILENAME, 'a') as f:
			f.write(str(acc) + '\n')
		f.close()


	"""
	The main method handling MDC logic (using single ensemble).
	"""
	def start(self, datasetName):
		#Get initial data buffer
		self.SInitialDataBuffer= self.source.initialData
		self.TInitialDataBuffer= self.target.initialData

		Properties.logger.info('Initializing Ensemble ...')
		#source model
		self.ensemble.generateNewModel(self.SInitialDataBuffer, self.TInitialDataBuffer, True)
		#target model
		self.ensemble.generateNewModel(self.SInitialDataBuffer, self.TInitialDataBuffer, False)
		Properties.logger.info(self.ensemble.getEnsembleSummary())

		sourceIndex = 0
		targetIndex = 0
		trueSourceNum = 0
		trueTargetNum = 0
		targetConfSum = 0

		Properties.logger.info('Starting MDC ...')
		while len(self.source.data) + len(self.target.data) > sourceIndex + targetIndex:
			ratio = (len(self.source.data) - sourceIndex) / (len(self.source.data) + len(self.target.data) - sourceIndex + targetIndex + 0.0)
			
			if (np.random.rand() <= ratio and sourceIndex < len(self.source.data)) or (targetIndex >= len(self.target.data) and sourceIndex < len(self.source.data)):
				sdata = self.source.data[sourceIndex]
				self.SDataBuffer.append(sdata)
				resSource = self.ensemble.evaluateEnsemble(sdata, True)
				self.SWindow.append(resSource[0])  # prediction of 0 or 1
				print('S', end="")
				# get Source Accuracy
				sourceIndex += 1
				trueSourceNum += resSource[0]
			elif targetIndex < len(self.target.data):
				tdata = self.target.data[targetIndex]
				self.TDataBuffer.append(tdata)
				resTarget = self.ensemble.evaluateEnsemble(tdata, False)
				conf = resTarget[1]  # confidence
				targetIndex += 1
				print('T', end="")

				# If conf is very close to 0.0 or 1.0, beta probability might become zero, which can make problems in change detection. Handling this scenario.
				if conf < 0.1:
					self.TWindow.append(0.1)
				elif conf > 0.995:
					self.TWindow.append(0.995)
				else:
					self.TWindow.append(resTarget[1])
				self.TPredictWindow.append(resTarget[0])

				#get Target Accuracy
				if resTarget[0] == tdata[-1]:
					trueTargetNum += 1
				acc = float(trueTargetNum)/(targetIndex)
				self.__saveResult(acc, datasetName)

				#save confidence
				targetConfSum += conf
				self.__saveResult(float(targetConfSum)/(targetIndex), datasetName+'_confidence')

			#Drift detection
			start = time.time()
			# srcCP = self.__detectDrift(self.SWindow, 0)
			# trgCP = self.__detectDrift(self.TWindow, 1)
			srcCP = self.__detectDriftJava(self.SWindow, 0)
			trgCP = self.__detectDriftJava(self.TWindow, 1)
			end = time.time()
			# print(int(end - start), end="")

			if srcCP != -1:
				self.__saveResult(5555555.0, datasetName+'_confidence')
				Properties.logger.info('-------------------------- S O U R C E		D R I F T ------------------------------------')
				Properties.logger.info('\nDrift found on source stream.')
				Properties.logger.info('dataIndex=' + str((targetIndex+sourceIndex)) + '\tsrcCP=' + str(srcCP) + '\ttrgCP=' + str(trgCP))

				#remove data from buffer till srcCP
				for i in xrange(srcCP):
					del self.SDataBuffer[0]
					del self.SWindow[0]

				#Exception with srcCP=0 (windowsize hit max or avg error is less than cutoff).
				#Keep atleast cushion number of instances
				if srcCP == 0:
					while len(self.SDataBuffer) > Properties.CUSHION:
						del self.SDataBuffer[0]
						del self.SWindow[0]

				Properties.logger.info('Instances left in source sliding window : ' + str(len(self.SDataBuffer)))
				Properties.logger.info('Instances left in target sliding window : ' + str(len(self.TDataBuffer)))

				Properties.logger.info('Updating ensemble weights')
				self.ensemble.updateWeight(self.SDataBuffer, True)

				Properties.logger.info('Training a model for source stream')
				self.ensemble.generateNewModel(self.SDataBuffer, self.TDataBuffer, True)
				Properties.logger.info(self.ensemble.getEnsembleSummary())


			if trgCP != -1:
				self.__saveResult(7777777.0, datasetName+'_confidence')
				Properties.logger.info('-------------------------- T A R G E T 	D R I F T ------------------------------------')
				Properties.logger.info('Drift found on target stream.')
				Properties.logger.info('dataIndex=' + str((targetIndex+sourceIndex)) + '\tsrcCP=' + str(srcCP) + '\ttrgCP=' + str(trgCP))

				#remove data from buffer till trgCP
				for i in xrange(trgCP):
					del self.TDataBuffer[0]
					del self.TWindow[0]
					del self.TPredictWindow[0]

				#Exception with trgCP=0 (windowsize hit max or avg error is less than cutoff).
				#Keep atleast cushion number of instances
				if trgCP == 0:
					while len(self.TDataBuffer) > Properties.CUSHION:
						del self.TDataBuffer[0]
						del self.TWindow[0]
						del self.TPredictWindow[0]

				Properties.logger.info('Instances left in source sliding window : ' + str(len(self.SDataBuffer)))
				Properties.logger.info('Instances left in target sliding window : ' + str(len(self.TDataBuffer)))

				Properties.logger.info('Updating ensemble weights')
				self.ensemble.updateWeight(self.TDataBuffer, False)

				if (len(self.SDataBuffer) > 0 and len(self.TDataBuffer)> 0):
					Properties.logger.info('Training a model for target stream')
					self.ensemble.generateNewModel(self.SDataBuffer, self.TDataBuffer, False)
					Properties.logger.info(self.ensemble.getEnsembleSummary())

			if (targetIndex+sourceIndex)%100 == 0:
				print('')

		Properties.logger.info('Done !!')
		return float(trueSourceNum)/(sourceIndex), float(trueTargetNum)/(targetIndex)


	"""
	Main module for MDC2 logic (using two separate ensembles)
	"""
	def start2(self, datasetName):
		#Get initial data buffer
		self.SInitialDataBuffer= self.source.initialData
		self.TInitialDataBuffer= self.target.initialData

		#Initialize Ensembles
		srcEnsemble = Ensemble(Properties.ENSEMBLE_SIZE)
		trgEnsemble = Ensemble(Properties.ENSEMBLE_SIZE)

		Properties.logger.info('Initializing Ensemble ...')
		#source model
		srcEnsemble.generateNewModel(self.SInitialDataBuffer, self.TInitialDataBuffer, True)
		Properties.logger.info('Source Ensemble')
		Properties.logger.info(srcEnsemble.getEnsembleSummary())
		#target model
		trgEnsemble.generateNewModel(self.SInitialDataBuffer, self.TInitialDataBuffer, False)
		Properties.logger.info('Target Ensemble')
		Properties.logger.info(trgEnsemble.getEnsembleSummary())

		dataIndex = 0
		trueTargetNum = 0
		targetConfSum = 0

		Properties.logger.info('Starting MDC2 ...')
		while(len(self.source.data) > dataIndex):
			print('.', end="")

			#Source Stream
			sdata = self.source.data[dataIndex]
			self.SDataBuffer.append(sdata)
			resSource = srcEnsemble.evaluateEnsemble(sdata, True)
			self.SWindow.append(resSource[0]) #prediction of 0 or 1

			#Target Stream
			tdata = self.target.data[dataIndex]
			self.TDataBuffer.append(tdata)
			resTarget = trgEnsemble.evaluateEnsemble(tdata, False)
			conf = resTarget[1] #confidence

			# If conf is very close to 0.0 or 1.0, beta probability might become zero, which can make problems in change detection. Handling this scenario.
			if conf < 0.1:
				self.TWindow.append(0.1)
			elif conf > 0.995:
				self.TWindow.append(0.995)
			else:
				self.TWindow.append(resTarget[1])
			self.TPredictWindow.append(resTarget[0])

			#get Target Accuracy
			if resTarget[0] == tdata[-1]:
				trueTargetNum += 1
			acc = float(trueTargetNum)/(dataIndex + 1)
			self.__saveResult(acc, datasetName)

			#save confidence
			targetConfSum += conf
			self.__saveResult(float(targetConfSum)/(dataIndex+1), datasetName+'_confidence')

			#Drift detection
			start = time.time()
			# srcCP = self.__detectDrift(self.SWindow, 0)
			# trgCP = self.__detectDrift(self.TWindow, 1)
			srcCP = self.__detectDriftJava(self.SWindow, 0)
			trgCP = self.__detectDriftJava(self.TWindow, 1)
			end = time.time()
			# print(int(end - start), end="")

			if srcCP != -1:
				self.__saveResult(5555555.0, datasetName+'_confidence')
				Properties.logger.info('-------------------------- S O U R C E		D R I F T ------------------------------------')
				Properties.logger.info('\nDrift found on source stream.')
				Properties.logger.info('dataIndex=' + str(dataIndex) + '\tsrcCP=' + str(srcCP) + '\ttrgCP=' + str(trgCP))

				#remove data from buffer till srcCP
				for i in xrange(srcCP):
					del self.SDataBuffer[0]
					del self.SWindow[0]

				#Exception with srcCP=0 (windowsize hit max or avg error is less than cutoff).
				#Keep atleast cushion number of instances
				if srcCP == 0:
					while len(self.SDataBuffer) > Properties.CUSHION:
						del self.SDataBuffer[0]
						del self.SWindow[0]

				Properties.logger.info('Instances left in source sliding window : ' + str(len(self.SDataBuffer)))
				Properties.logger.info('Instances left in target sliding window : ' + str(len(self.TDataBuffer)))

				#Updating source Ensemble
				Properties.logger.info('Updating source ensemble weights')
				srcEnsemble.updateWeight(self.SDataBuffer, True)

				Properties.logger.info('Training a model for source stream')
				srcEnsemble.generateNewModel(self.SDataBuffer, self.TDataBuffer, True)
				Properties.logger.info('Source Ensemble')
				Properties.logger.info(srcEnsemble.getEnsembleSummary())


			if trgCP != -1:
				self.__saveResult(7777777.0, datasetName+'_confidence')
				Properties.logger.info('-------------------------- T A R G E T 	D R I F T ------------------------------------')
				Properties.logger.info('Drift found on target stream.')
				Properties.logger.info('dataIndex=' + str(dataIndex) + '\tsrcCP=' + str(srcCP) + '\ttrgCP=' + str(trgCP))

				#remove data from buffer till trgCP
				for i in xrange(trgCP):
					del self.TDataBuffer[0]
					del self.TWindow[0]
					del self.TPredictWindow[0]

				#Exception with trgCP=0 (windowsize hit max or avg error is less than cutoff).
				#Keep atleast cushion number of instances
				if trgCP == 0:
					while len(self.TDataBuffer) > Properties.CUSHION:
						del self.TDataBuffer[0]
						del self.TWindow[0]
						del self.TPredictWindow[0]

				Properties.logger.info('Instances left in source sliding window : ' + str(len(self.SDataBuffer)))
				Properties.logger.info('Instances left in target sliding window : ' + str(len(self.TDataBuffer)))

				Properties.logger.info('Updating target ensemble weights')
				trgEnsemble.updateWeight(self.TDataBuffer, False)

				Properties.logger.info('Training a model for target stream')
				trgEnsemble.generateNewModel(self.SDataBuffer, self.TDataBuffer, False)
				Properties.logger.info('Target Ensemble')
				Properties.logger.info(trgEnsemble.getEnsembleSummary())

			dataIndex += 1
			if dataIndex%100 == 0:
				print('')

		Properties.logger.info('Done !!')


	"""
	Baseline skmm (single target model with initial train only)
	"""
	def start_skmm(self, datasetName):
		#Get initial data buffer
		self.SInitialDataBuffer= self.source.initialData
		self.TInitialDataBuffer= self.target.initialData

		#Initialize Model
		model = Model()
		model.train(self.SInitialDataBuffer, self.TInitialDataBuffer, Properties.MAXVAR)

		dataIndex = 0
		trueTargetNum = 0

		Properties.logger.info('Starting skmm baseline ...')
		while(len(self.source.data) > dataIndex):
			print('.', end="")

			#Source Stream
			sdata = self.source.data[dataIndex]
			self.SDataBuffer.append(sdata)

			#Target Stream
			tdata = self.target.data[dataIndex]
			self.TDataBuffer.append(tdata)

			#test data instance in each model

			resTarget = model.test([tdata], Properties.MAXVAR)

			#get Target Accuracy
			if resTarget[0][0] == tdata[-1]:
				trueTargetNum += 1
			acc = float(trueTargetNum)/(dataIndex + 1)
			self.__saveResult(acc, datasetName)

			dataIndex += 1
			if dataIndex%100 == 0:
				print('')

		Properties.logger.info('Done !!')


	"""
	Baseline mkmm (single target model trained periodically)
	"""
	def start_mkmm(self, datasetName):
		#Get initial data buffer
		self.SInitialDataBuffer= self.source.initialData
		self.TInitialDataBuffer= self.target.initialData

		#Initialize Model
		model = Model()
		model.train(self.SInitialDataBuffer, self.TInitialDataBuffer, Properties.MAXVAR)

		dataIndex = 0
		trueTargetNum = 0

		Properties.logger.info('Starting skmm baseline ...')
		while(len(self.source.data) > dataIndex):
			print('.', end="")

			#Source Stream
			sdata = self.source.data[dataIndex]
			self.SDataBuffer.append(sdata)

			#Target Stream
			tdata = self.target.data[dataIndex]
			self.TDataBuffer.append(tdata)

			#test data instance in each model
			resTarget = model.test([tdata], Properties.MAXVAR)

			#get Target Accuracy
			if resTarget[0][0] == tdata[-1]:
				trueTargetNum += 1
			acc = float(trueTargetNum)/(dataIndex + 1)
			self.__saveResult(acc, datasetName)

			dataIndex += 1
			if dataIndex%100 == 0:
				print('')
			if dataIndex%Properties.MAX_WINDOW_SIZE == 0:
				model = Model()
				model.train(self.SDataBuffer, self.TDataBuffer, Properties.MAXVAR)
				self.SDataBuffer = []
				self.TDataBuffer = []

		Properties.logger.info('Done !!')


	"""
	Baseline srconly using an ensemble of only source classifiers.
	Target labels predicted from this ensemble using its target weights.
	"""
	def start_srconly(self, datasetName):
		#Get initial data buffer
		self.SInitialDataBuffer= self.source.initialData
		self.TInitialDataBuffer= self.target.initialData

		#Initialize Ensembles
		srcEnsemble = Ensemble(Properties.ENSEMBLE_SIZE)

		Properties.logger.info('Initializing Ensemble ...')
		#source model
		srcEnsemble.generateNewModel(self.SInitialDataBuffer, self.TInitialDataBuffer, True)
		Properties.logger.info('Source Ensemble')
		Properties.logger.info(srcEnsemble.getEnsembleSummary())

		dataIndex = 0
		trueTargetNum = 0
		targetConfSum = 0

		Properties.logger.info('Starting srconly-MDC ...')
		while(len(self.source.data) > dataIndex):
			print('.', end="")

			#Source Stream
			sdata = self.source.data[dataIndex]
			self.SDataBuffer.append(sdata)
			resSource = srcEnsemble.evaluateEnsemble(sdata, True)
			self.SWindow.append(resSource[0]) #prediction of 0 or 1

			#Target Stream
			tdata = self.target.data[dataIndex]
			self.TDataBuffer.append(tdata)
			resTarget = srcEnsemble.evaluateEnsemble(tdata, False)
			conf = resTarget[1] #confidence

			# If conf is very close to 0.0 or 1.0, beta probability might become zero, which can make problems in change detection. Handling this scenario.
			if conf < 0.1:
				self.TWindow.append(0.1)
			elif conf > 0.995:
				self.TWindow.append(0.995)
			else:
				self.TWindow.append(resTarget[1])
			self.TPredictWindow.append(resTarget[0])

			#get Target Accuracy
			if resTarget[0] == tdata[-1]:
				trueTargetNum += 1
			acc = float(trueTargetNum)/(dataIndex + 1)
			self.__saveResult(acc, datasetName)

			#save confidence
			targetConfSum += conf
			self.__saveResult(float(targetConfSum)/(dataIndex+1), datasetName+'_confidence')

			#Drift detection
			start = time.time()
			# srcCP = self.__detectDrift(self.SWindow, 0)
			# trgCP = self.__detectDrift(self.TWindow, 1)
			srcCP = self.__detectDriftJava(self.SWindow, 0)
			trgCP = self.__detectDriftJava(self.TWindow, 1)
			end = time.time()
			# print(int(end - start), end="")

			if srcCP != -1:
				self.__saveResult(5555555.0, datasetName+'_confidence')
				Properties.logger.info('-------------------------- S O U R C E		D R I F T ------------------------------------')
				Properties.logger.info('\nDrift found on source stream.')
				Properties.logger.info('dataIndex=' + str(dataIndex) + '\tsrcCP=' + str(srcCP) + '\ttrgCP=' + str(trgCP))

				#remove data from buffer till srcCP
				for i in xrange(srcCP):
					del self.SDataBuffer[0]
					del self.SWindow[0]

				#Exception with srcCP=0 (windowsize hit max or avg error is less than cutoff).
				#Keep atleast cushion number of instances
				if srcCP == 0:
					while len(self.SDataBuffer) > Properties.CUSHION:
						del self.SDataBuffer[0]
						del self.SWindow[0]

				Properties.logger.info('Instances left in source sliding window : ' + str(len(self.SDataBuffer)))
				Properties.logger.info('Instances left in target sliding window : ' + str(len(self.TDataBuffer)))

				#Updating source Ensemble
				Properties.logger.info('Updating source ensemble weights')
				srcEnsemble.updateWeight(self.SDataBuffer, True)

				Properties.logger.info('Training a model for source stream')
				srcEnsemble.generateNewModel(self.SDataBuffer, self.TDataBuffer, True)
				Properties.logger.info('Source Ensemble')
				Properties.logger.info(srcEnsemble.getEnsembleSummary())


			if trgCP != -1:
				self.__saveResult(7777777.0, datasetName+'_confidence')
				Properties.logger.info('-------------------------- T A R G E T 	D R I F T ------------------------------------')
				Properties.logger.info('Drift found on target stream.')
				Properties.logger.info('dataIndex=' + str(dataIndex) + '\tsrcCP=' + str(srcCP) + '\ttrgCP=' + str(trgCP))

				#remove data from buffer till trgCP
				for i in xrange(trgCP):
					del self.TDataBuffer[0]
					del self.TWindow[0]
					del self.TPredictWindow[0]

				#Exception with trgCP=0 (windowsize hit max or avg error is less than cutoff).
				#Keep atleast cushion number of instances
				if trgCP == 0:
					while len(self.TDataBuffer) > Properties.CUSHION:
						del self.TDataBuffer[0]
						del self.TWindow[0]
						del self.TPredictWindow[0]

				Properties.logger.info('Instances left in source sliding window : ' + str(len(self.SDataBuffer)))
				Properties.logger.info('Instances left in target sliding window : ' + str(len(self.TDataBuffer)))

				Properties.logger.info('Updating target ensemble weights')
				srcEnsemble.updateWeight(self.TDataBuffer, False)

				Properties.logger.info('Training a model for target stream')
				srcEnsemble.generateNewModel(self.SDataBuffer, self.TDataBuffer, True)
				Properties.logger.info('Target Ensemble')
				Properties.logger.info(srcEnsemble.getEnsembleSummary())


			dataIndex += 1
			if dataIndex%100 == 0:
				print('')

		Properties.logger.info('Done !!')


	"""
	Baseline trgonly using an ensemble of only target classifiers.
	Target labels predicted from this ensemble using its target weights.
	Source drift is computed using source-weighted ensemble prediction.
	"""
	def start_trgonly(self, datasetName):
		#Get initial data buffer
		self.SInitialDataBuffer= self.source.initialData
		self.TInitialDataBuffer= self.target.initialData

		#Initialize Ensembles
		trgEnsemble = Ensemble(Properties.ENSEMBLE_SIZE)

		Properties.logger.info('Initializing Ensemble ...')
		#target model
		trgEnsemble.generateNewModel(self.SInitialDataBuffer, self.TInitialDataBuffer, False)
		Properties.logger.info('Target Ensemble')
		Properties.logger.info(trgEnsemble.getEnsembleSummary())

		dataIndex = 0
		trueTargetNum = 0
		targetConfSum = 0

		Properties.logger.info('Starting trgonly-MDC ...')
		while(len(self.source.data) > dataIndex):
			print('.', end="")

			#Source Stream
			sdata = self.source.data[dataIndex]
			self.SDataBuffer.append(sdata)
			resSource = trgEnsemble.evaluateEnsemble(sdata, True)
			self.SWindow.append(resSource[0]) #prediction of 0 or 1

			#Target Stream
			tdata = self.target.data[dataIndex]
			self.TDataBuffer.append(tdata)
			resTarget = trgEnsemble.evaluateEnsemble(tdata, False)
			conf = resTarget[1] #confidence

			# If conf is very close to 0.0 or 1.0, beta probability might become zero, which can make problems in change detection. Handling this scenario.
			if conf < 0.1:
				self.TWindow.append(0.1)
			elif conf > 0.995:
				self.TWindow.append(0.995)
			else:
				self.TWindow.append(resTarget[1])
			self.TPredictWindow.append(resTarget[0])

			#get Target Accuracy
			if resTarget[0] == tdata[-1]:
				trueTargetNum += 1
			acc = float(trueTargetNum)/(dataIndex + 1)
			self.__saveResult(acc, datasetName)

			#save confidence
			targetConfSum += conf
			self.__saveResult(float(targetConfSum)/(dataIndex+1), datasetName+'_confidence')

			#Drift detection
			start = time.time()
			# srcCP = self.__detectDrift(self.SWindow, 0)
			# trgCP = self.__detectDrift(self.TWindow, 1)
			srcCP = self.__detectDriftJava(self.SWindow, 0)
			trgCP = self.__detectDriftJava(self.TWindow, 1)
			end = time.time()
			# print(int(end - start), end="")

			if srcCP != -1:
				self.__saveResult(5555555.0, datasetName+'_confidence')
				Properties.logger.info('-------------------------- S O U R C E		D R I F T ------------------------------------')
				Properties.logger.info('\nDrift found on source stream.')
				Properties.logger.info('dataIndex=' + str(dataIndex) + '\tsrcCP=' + str(srcCP) + '\ttrgCP=' + str(trgCP))

				#remove data from buffer till srcCP
				for i in xrange(srcCP):
					del self.SDataBuffer[0]
					del self.SWindow[0]

				#Exception with srcCP=0 (windowsize hit max or avg error is less than cutoff).
				#Keep atleast cushion number of instances
				if srcCP == 0:
					while len(self.SDataBuffer) > Properties.CUSHION:
						del self.SDataBuffer[0]
						del self.SWindow[0]

				Properties.logger.info('Instances left in source sliding window : ' + str(len(self.SDataBuffer)))
				Properties.logger.info('Instances left in target sliding window : ' + str(len(self.TDataBuffer)))

				#Updating source Ensemble
				Properties.logger.info('Updating source ensemble weights')
				trgEnsemble.updateWeight(self.SDataBuffer, True)

				Properties.logger.info('Training a model for source stream')
				trgEnsemble.generateNewModel(self.SDataBuffer, self.TDataBuffer, False)
				Properties.logger.info('Source Ensemble')
				Properties.logger.info(trgEnsemble.getEnsembleSummary())


			if trgCP != -1:
				self.__saveResult(7777777.0, datasetName+'_confidence')
				Properties.logger.info('-------------------------- T A R G E T 	D R I F T ------------------------------------')
				Properties.logger.info('Drift found on target stream.')
				Properties.logger.info('dataIndex=' + str(dataIndex) + '\tsrcCP=' + str(srcCP) + '\ttrgCP=' + str(trgCP))

				#remove data from buffer till trgCP
				for i in xrange(trgCP):
					del self.TDataBuffer[0]
					del self.TWindow[0]
					del self.TPredictWindow[0]

				#Exception with trgCP=0 (windowsize hit max or avg error is less than cutoff).
				#Keep atleast cushion number of instances
				if trgCP == 0:
					while len(self.TDataBuffer) > Properties.CUSHION:
						del self.TDataBuffer[0]
						del self.TWindow[0]
						del self.TPredictWindow[0]

				Properties.logger.info('Instances left in source sliding window : ' + str(len(self.SDataBuffer)))
				Properties.logger.info('Instances left in target sliding window : ' + str(len(self.TDataBuffer)))

				Properties.logger.info('Updating target ensemble weights')
				trgEnsemble.updateWeight(self.TDataBuffer, False)

				Properties.logger.info('Training a model for target stream')
				trgEnsemble.generateNewModel(self.SDataBuffer, self.TDataBuffer, False)
				Properties.logger.info('Target Ensemble')
				Properties.logger.info(trgEnsemble.getEnsembleSummary())

			dataIndex += 1
			if dataIndex%100 == 0:
				print('')

		Properties.logger.info('Done !!')
