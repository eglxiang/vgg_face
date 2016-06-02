# vgg_face
This is a test script for the VGG_face deep model. It simply compares the correlation between two deeply learned features corresponding with two testing facial images needed to be verified.

Face recognition seems rising again. Google, Facebook and SenseTime have almost solved it on benchmarks such as Labelled Face in the Wild (LFW). Their face recognizer are known  as FaceNet, DeepFace and FaceID, respectively. Andrew Zisserman's group from the University of Oxford released their deep face model called VGG Face Descriptor, which is claimed to be trained on millions of  face images. Today I take it and give it a shot on the task of face verification - determining whether a pair of two face images are from the same person or not.  I simply compare the correlation (or cosine-similarity) between two deeply learned features corresponding with the two testing facial images that needed to be verified. Below are some results on LFW. Does the similarity (the third column) really tell you if the left person and the right one are the same person?

Depending on the operating system, you may set link libraries and include headfiles in the following way.

/usr/lib64/libboost_system.so

/usr/local/lib/libopencv_highgui.so

/usr/local/lib/libopencv_core.so

/usr/local/lib/libopencv_imgproc.so

/usr/lib64/libglog.so

/usr/local/lib/libcaffe.so

/usr/local/lib/libopencv_imgcodecs.so

Linker:

/usr/local/lib

/usr/lib

Compiler:

/usr/local/include
