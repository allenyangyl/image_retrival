import numpy as np

train_listfile = 'TrainImages.txt'
train_list = np.loadtxt(train_listfile, str, comments=None, delimiter='\n')
test_listfile = 'TestImages.txt'
test_list = np.loadtxt(test_listfile, str, comments=None, delimiter='\n')
label_listfile = 'label_names.txt'
label_list = np.loadtxt(label_listfile, str, comments=None, delimiter='\n')

label_name = dict()
for i, label in enumerate(label_list):
	label_name[label] = i

train_labelfile = 'TrainLabels.txt'
f_train = open(train_labelfile, 'w')
for train_data in train_list:
	label = train_data.split('/')[0]
	f_train.write(train_data + ' ' + str(label_name[label]) + '\n')

test_labelfile = 'TestLabels.txt'
f_test = open(test_labelfile, 'w')
for test_data in test_list:
	label = test_data.split('/')[0]
	f_test.write(test_data + ' ' + str(label_name[label]) + '\n')
