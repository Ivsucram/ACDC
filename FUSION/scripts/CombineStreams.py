import random
import sys

sf = open(sys.argv[1])
tf = open(sys.argv[2])
of = open(sys.argv[3], mode='w')
numLines = 0
while True:
	randomNum = random.uniform(0,1)
	if randomNum < 0.5:
		sl = sf.readline()
		if not sl:
			break
		of.write(sl)
	else:
		tl = tf.readline()
		if not tl:
			break
		of.write(tl)
	numLines += 1
	if numLines % 1000 == 0:
		print("Process Lines: ", numLines)
for line in sf:
	of.write(line)
for line in tf:
	of.write(line)
sf.close()
tf.close()
of.close()
