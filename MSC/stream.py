
class Stream(object):

	data = None
	initialData = None

	"""
	Initialize a stream by reading data from file.
	Input data file formats: ARFF or Sparse (.data)
	"""
	def __init__(self, filename, classList, initialSize):
		self.data = []
		self.initialData = []
		self.MAXVAR = self.__readData(filename, classList, initialSize)


	"""
	Read data from file in CSV or Sparse format.
	Return maximum number of variables.
	"""
	def __readData(self, filename, classList, initialSize):
		with open(filename) as f:
			data = f.readlines()

		maxvar = 0
		for i in data:
			d = {}
			if filename.endswith('.csv'):
				features = i.strip().split(',')
				if features[-1] not in classList:
					classList.append(features[-1])
				d[-1] = float(classList.index(features[-1]))
				for j in xrange(len(features)-1):
					d[j] = float(features[j])
				maxvar = len(features)-1
			else:
				features = i.strip().split(' ')
				for fea in features:
					val = fea.strip().split(':')
					if len(val) < 2:
						d[-1] = float(val[0])
					else:
						d[int(val[0])-1] = float(val[1])
					#get maximum number of features
					if maxvar < int(val[0]):
						maxvar = int(val[0])

			if len(self.initialData) < initialSize:
				self.initialData.append(d)
			else:
				self.data.append(d)

		return maxvar
