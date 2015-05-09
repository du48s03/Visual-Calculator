import os, sys
includespath = os.path.abspath('../includes')
sys.path.insert(0, includespath)
import posture
import cPickle
import re
import numpy as np


datasetname = sys.argv[1]


datafile = open(datasetname, 'r')
lists = cPickle.load(datafile)
data, labels = (lists[0], lists[1])

datafile.close()

data_good = np.ones(len(labels), dtype=bool)
for i in xrange(len(labels)):    
    if labels[i] not in posture.poses.values():
        data_good[i] = False

data = data[data_good]
labels = labels[data_good]
print "Good data = ", len(data)


test_data = []
test_label = []
rand_ind = np.random.permutation(range(len(labels)))
test_data = data[rand_ind[:int(0.2*len(labels))]]
test_label = labels[rand_ind[:int(0.2*len(labels))]]
train_data = data[rand_ind[int(0.2*len(labels))+1:]]
train_label = labels[rand_ind[int(0.2*len(labels))+1:]]

print len(train_data)
print len(test_data)

srch = re.search(r'^(.+)\.([^\.]*)$', datasetname)
filename_handle = srch.group(1)
ext  = srch.group(2)

f1 = open(filename_handle+'_train'+'.'+ext, 'w')
f2 = open(filename_handle+'_test'+'.'+ext, 'w')
cPickle.dump((train_data, train_label), f1)
cPickle.dump((test_data, test_label), f2)
f1.close()
f2.close()