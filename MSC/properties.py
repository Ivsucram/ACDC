import logging, subprocess
import threading, random


class Properties(object):
	GAMMA = 0.0
	CUSHION = 0
	SENSITIVITY = 0.0
	MAX_WINDOW_SIZE = 0
	ENSEMBLE_SIZE = 0
	CONFTHRESHOLD = 0.0
	CONFCUTOFF = 0.0
	INITIAL_DATA_SIZE = 0
	MAXVAR = 0

	IDENTIFIER = ''
	OUTFILENAME = ''
	TEMPDIR = ''
	LOGFILE = ''

	BASEDIR = ''
	SRCAPPEND = ''
	TRGAPPEND = ''

	PY4JPORT = 25333

	logger = None

	def __init__(self, propfilename, datasetName):
		dict = {}
		with open(propfilename) as f:
			for line in f:
				(key,val) = line.split('=')
				dict[key.strip()] = val.strip()

		self.__class__.GAMMA = float(dict['gamma'])
		self.__class__.CUSHION = int(dict['cushion'])
		self.__class__.SENSITIVITY = float(dict['sensitivity'])
		self.__class__.MAX_WINDOW_SIZE = int(dict['maxWindowSize'])
		self.__class__.ENSEMBLE_SIZE = int(dict['ensemble_size'])
		self.__class__.CONFTHRESHOLD = float(dict['confthreshold'])
		self.__class__.CONFCUTOFF = float(dict['confcutoff'])
		self.__class__.INITIAL_DATA_SIZE = int(dict['initialDataSize'])

		self.__class__.IDENTIFIER = datasetName + '_' + str(self.__class__.MAX_WINDOW_SIZE)
		self.__class__.OUTFILENAME = self.__class__.IDENTIFIER + '_' + dict['output_file_name']
		self.__class__.TEMPDIR = dict['tempDir']
		self.__class__.LOGFILE = self.__class__.IDENTIFIER + '_' + dict['logfile']

		if self.__class__.logger: self.__class__.logger = None
		self.__class__.logger = self.__setupLogger()

		self.__class__.MAXVAR = 0

		self.__class__.BASEDIR = dict['baseDir']
		self.__class__.SRCAPPEND = dict['srcfileAppend']
		self.__class__.TRGAPPEND = dict['trgfileAppend']

		self.__class__.PY4JPORT = random.randint(25333, 30000)

		t = threading.Thread(target=self.__startCPDJava)
		t.daemon = True
		t.start()



	def __startCPDJava(self):
		subprocess.call(['java', '-jar', 'change_point.jar', str(self.__class__.GAMMA), str(self.__class__.SENSITIVITY), str(self.__class__.MAX_WINDOW_SIZE), str(self.__class__.CUSHION), str(self.__class__.CONFCUTOFF), str(self.__class__.PY4JPORT)])



	def __setupLogger(self):
		logger = logging.getLogger(__name__)
		logger.setLevel(logging.INFO)

		sh = logging.StreamHandler()
		sh.setLevel(logging.INFO)
		logger.addHandler(sh)
		handler = logging.FileHandler(self.__class__.LOGFILE)
		handler.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		return logger



	def summary(self):
		line = 'Parameter values are as follows:'
		line += '\nGamma = ' + str(self.GAMMA)
		line += '\nSensitivity = ' + str(self.SENSITIVITY)
		line += '\nEnsemble Size = ' + str(self.ENSEMBLE_SIZE)
		line += '\nConfidence Threshold (NOT USED) = ' + str(self.CONFTHRESHOLD)
		line += '\nConfidence Cutoff = ' + str(self.CONFCUTOFF)
		line += '\nMax Window Size = ' + str(self.MAX_WINDOW_SIZE)
		line += '\nInitial Training Size = ' + str(self.INITIAL_DATA_SIZE)
		line += '\nMaximum Num Variables = ' + str(self.MAXVAR)
		line += '\nOutput File = ' + str(self.OUTFILENAME)
		return line