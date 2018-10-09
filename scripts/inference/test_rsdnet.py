import numpy as np
import os.path
import os
import json
import scipy
import argparse
import math
from sklearn.preprocessing import normalize
from scipy import misc

caffe_root = 'PATH_TO_YOUR_CAFFE/caffe-rsdnet/'  # # MODIFY PATH for YOUR SETTING

import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()
caffe.set_device(0)   # Set the gpu_id 

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)


fname = './data/test.txt'
save_folder = './predictions/saliency_maps_pascals_rsdnet_1/'

if not os.path.exists(save_folder): 
   os.makedirs(save_folder)
   print ('Save directory created!')


with open(fname) as f:
    labelFiles = f.read().splitlines()

for i in range(0, args.iter):

    net.forward()

    image = net.blobs['data'].data
    label_net = net.blobs['label'].data
    
    labelFile = labelFiles[i].split(' ')[1]
    
    
    label = misc.imread(labelFile)
    predicted = net.blobs['predicted_saliency_mask_interp'].data
    image = np.squeeze(image[0,:,:,:])
    output = np.squeeze(predicted[0,:,:,:])
    
    ind = output[0:label.shape[0], 0:label.shape[1]]
    
    labelname = labelFile.split('gt/')

    misc.toimage(ind,cmin = 0.0, cmax = 255).save( save_folder + labelname[1])
   
    print 'Processed: ', labelname[1]

print 'Success!'

