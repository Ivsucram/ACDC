import sys
from manager import Manager
from properties import Properties
import time



def main(datasetName):
	# datasetName = 'powersupply_normalized'

	props = Properties('config.properties', datasetName)
	srcfile = Properties.BASEDIR + datasetName + Properties.SRCAPPEND
	trgfile = Properties.BASEDIR + datasetName + Properties.TRGAPPEND
	mgr = Manager(srcfile, trgfile)

	Properties.logger.info(props.summary())
	Properties.logger.info('Start Stream Simulation')

	start_time = time.time()
	source_cr, target_cr = mgr.start(datasetName)
	training_time = time.time() - start_time
	# mgr.start2(datasetName)

	#baseline methods
	# mgr.start_skmm(datasetName)
	# mgr.start_mkmm(datasetName)
	# mgr.start_srconly(datasetName)
	# mgr.start_trgonly(datasetName)

	mgr.gateway.shutdown()
	return {'SourceCR': source_cr, 'TargetCR': target_cr, 'TrainingTime': training_time}
