import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '../../caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import time

t0 = time.time()
test_listfile = 'TestLabels.txt'
test_list = np.loadtxt(test_listfile, str, comments=None, delimiter='\n')
data_counts = len(test_list)
#query_file = [test_list[int(data_counts/4)].split(' ')[0], 
#		test_list[int(data_counts/2)].split(' ')[0], 
#		test_list[int(3*data_counts/4)].split(' ')[0]]
query_label = [7, 3, 16, 29, 56]
directory = 'Images/'

caffe.set_device(0)
caffe.set_mode_gpu()
#net = caffe.Net('/home/yiliny1/caffe/models/bvlc_reference_caffenet/deploy.prototxt', 
#		'/home/yiliny1/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', 
#		caffe.TEST)
net = caffe.Net('/home/yiliny1/bosch/MIT67_from_ImageNet/deploy.prototxt',
                '/home/yiliny1/bosch/MIT67_from_ImageNet/model_iter_30000.caffemodel',
                caffe.TEST)

batch_size = net.blobs['data'].data.shape[0]
batch_count = int(np.ceil(data_counts * 1.0 / batch_size))

blob = caffe.proto.caffe_pb2.BlobProto()
mu = open('/home/yiliny1/caffe/data/ilsvrc12/imagenet_mean.binaryproto', "rb").read()
blob.ParseFromString(mu)
mu = caffe.io.blobproto_to_array(blob)
mu = mu.mean(2).mean(2).reshape(3)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

t1 = time.time()
features = np.array([])
for file in query_file:
	image = caffe.io.load_image(directory + file)
	image = transformer.preprocess('data', image)
	net.blobs['data'].data[0] = image
	out = net.forward()
	fc7 = net.blobs['fc7'].data[0].flatten()
	try:
		features = np.vstack((features, fc7))
	except: 
		features = np.hstack((features, fc7))

score = np.array([])
for i in range(batch_count):
	print 'test batch: ' + str(i)

	for j in range(batch_size): 
		id = i * batch_size + j
		if id >= data_counts: 
			break

		fname = test_list[id].split(' ')[0]
		image = caffe.io.load_image(directory + fname)
		image = transformer.preprocess('data', image)
		net.blobs['data'].data[j] = image
	out = net.forward()

	for j in range(batch_size): 
		id = i * batch_size + j
		if id >= data_counts:
			break
		score_test = np.array([])
		fc7 = net.blobs['fc7'].data[j].flatten()
		for k in range(len(query_file)):
			score_this = np.mean(np.square(features[k] - fc7))
			score_test = np.hstack((score_test, score_this))
		try: 
			score = np.vstack((score, score_test))
		except:
			score = np.hstack((score, score_test))

t2 = time.time()
ind = np.argsort(score, axis = 0)
#print 'index: '
#print ind
score = np.sort(score, axis = 0)
#print '\nscore: '
#print score
t3 = time.time()

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
num = 20
nums = [1, 3, 5, 10, 20, 50]
save_directory = 'ranking/'
if not os.path.exists(save_directory):
	os.makedirs(save_directory)
for i in range(len(query_file)): 
	fname = query_file[i]
	image = mpimg.imread(directory + fname)
	plt.imsave(save_directory + 'img' + str(i) + '_query.png', image)
	#print 'img' + str(i) + '_query.png saved'
	for j in range(num): 
		fname = test_list[ind[j, i]].split(' ')[0]
		image = mpimg.imread(directory + fname)
		plt.imsave(save_directory + 'img' + str(i) + '_rank' + str(j+1) + '_score' + str(round(score[j, i], 4)) + '.png', image)
		#print 'img' + str(i) + '_rank' + str(j) + '_score' + str(round(score[j, i], 4)) + '.png saved'
	for n in range(len(nums)):
		accuracy = 0
		for j in range(nums[n]):
			lbl = test_list[ind[j, i]].split(' ')[1]
			if query_label[i] == int(lbl):
				accuracy = accuracy + 1
		accuracy = accuracy / float(nums[n])
		print 'file: ' + query_file[i] + ', top ' + str(nums[n]) + ', accuracy: ' + str(accuracy)
t4 = time.time()

print 'network set up time: '
print t1 - t0
print 'calculating time: '
print t2 - t1
print 'comparing time: '
print t3 - t2
print 'image saving time: '
print t4 - t3
