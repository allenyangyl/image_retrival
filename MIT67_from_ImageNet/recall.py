import numpy as np
import scipy.spatial.distance as dist
from numpy.linalg import inv

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

features = np.array([])
scores = np.zeros((data_counts, data_counts))
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
		try:
			features = np.vstack((features, fc7))
		except: 
			features = np.hstack((features, fc7))

print features
print features.shape
var = np.var(features, axis = 0)
#VI = inv(np.cov(features.T))

for i in range(data_counts):
	print i
	for j in range(data_counts):
		#score = np.mean(np.square(features[i] - features[j]))
		#score = dist.euclidean(features[i], features[j])
		score = dist.cosine(features[i], features[j])
		#score = dist.seuclidean(features[i], features[j], var)
		#score = dist.mahalanobis(features[i], features[j], VI)
		scores[i, j] = score
print scores
print scores.shape

ind = np.argsort(scores, axis = 0)
#print 'index: '
print ind
score = np.sort(scores, axis = 0)
#print '\nscore: '
print score

nums = [1, 3, 5, 10, 20, 50]
for n in range(len(nums)):
	print n
	accuracy = 0
	for i in range(data_counts):
		flag = 0
		query_lbl = int(test_list[i].split(' ')[1])
		for j in range(1, nums[n] + 1):
			lbl = int(test_list[ind[j, i]].split(' ')[1])
			if query_lbl == lbl:
				flag = 1
		accuracy = accuracy + flag
	accuracy = accuracy / float(data_counts)
	print 'top ' + str(nums[n]) + ', accuracy: ' + str(accuracy)
