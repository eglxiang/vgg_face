# Xiang Xiang (eglxiang@gmail.com), March 2016, MIT license.

import numpy as np
import cv2 
import caffe
import time

img = caffe.io.load_image( "ak.png" )
img = img[:,:,::-1]*255.0 # convert RGB->BGR
avg = np.array([129.1863,104.7624,93.5940])
img = img - avg # subtract mean (numpy takes care of dimensions :)

img = img.transpose((2,0,1)) 
img = img[None,:] # add singleton dimension

caffe.set_mode_cpu()
net = caffe.Net("VGG_FACE_deploy.prototxt","VGG_FACE.caffemodel",  caffe.TEST)

start_time = time.time()
out = net.forward_all( data = img )
elapsed_time = time.time() - start_time
print elapsed_time
