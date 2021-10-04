import sys
from manager import Manager
from properties import Properties
import time

"""
Parameters
datasetName: main part of dataset file name, e.g., powersupply for powersupply_source_stream.csv, powersupply_target_stream.csv
baseline: 1=startMscKLIEP, 2=start, 3=start2, 4=start_skmm, 5=start_mkmm, 6=start_srconly, 7=start_trgonly
"""
def main(datasetName, opt=1.0):
	props = Properties('config.properties', datasetName)
	srcfile = Properties.BASEDIR + datasetName + Properties.SRCAPPEND
	trgfile = Properties.BASEDIR + datasetName + Properties.TRGAPPEND
	mgr = Manager(srcfile, trgfile)

	Properties.logger.info(props.summary())
	Properties.logger.info('Start Stream Simulation')

	source_cr, target_cr, training_time = mgr.startFusion(datasetName, opt)

	print ""
	print {'SourceCR': source_cr, 'TargetCR': target_cr, 'TrainingTime': training_time}
	return {'SourceCR': source_cr, 'TargetCR': target_cr, 'TrainingTime': training_time}
"""
if __name__ == '__main__':
	main()
"""

