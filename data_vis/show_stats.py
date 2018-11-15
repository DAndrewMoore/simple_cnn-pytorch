
import numpy as np
import matplotlib.pyplot as plt

###############################
# Params
train_fp = 'train.csv'
###############################

def readTrainingSet(train_fp):
	target = open(train_fp, 'r')
	target.readline()

	data = []
	for line in target:
		clss = np.zeros(28)
		x = line.split(',')[1]
		x = x.split(' ')
		for num in x:
			num = int(num)
			clss[num] = 1
		data.append(clss)
	target.close()
	return data

def getClassCounts(data):
	class_count = np.zeros(28)
	for row in data:
		class_count += row
	return class_count

def showBarChart(counts):
	plt.bar(np.arange(28), counts)
	plt.show()

def getCombinationCounts(data):
	x = {}
	for row in data:
		ones_idcs = (row == 1).nonzero()[0]
		str_rep = '_'.join(list(map(str, ones_idcs)))
		try:
			tmp = x[str_rep]
		except:
			tmp = 0
		tmp += 1
		x[str_rep] = tmp
	return x

def getLongestCombination(kvPair):
	longest = 0
	for key in kvPair.keys():
		num = len(key.split('_'))
		if num > longest:
			longest = num
	return longest

def padOutput(proposedKey, num2match):
	num_cur = len(proposedKey.split('\t'))
	for i in range(num2match - num_cur):
		proposedKey += '\t'
	return proposedKey

def writeCombinationTsv(kvPair, fp):
	numTabs = getLongestCombination(kvPair)
	target = open(fp, 'w')
	for key in kvPair.keys():
		key_str = padOutput('\t'.join(key.split('_')), numTabs)
		target.write('%s\t%d\n' % (key_str, kvPair[key]))
	target.close()

class_base = np.arange(28)
data = readTrainingSet(train_fp)
counts = getClassCounts(data)
