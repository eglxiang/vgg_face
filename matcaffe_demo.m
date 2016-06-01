%  Copyright (c) 2015, Omkar M. Parkhi
%  All rights reserved.
% add path of Matlab such as '/usr/local/matlab'
% add path of Caffe Matlab such as '.../caffe/matlab/'
img = imread('ak.png');
img = single(img);

averageImg = [129.1863,104.7624,93.5940] ;

img = cat(3,img(:,:,1)-averageImg(1),...
    img(:,:,2)-averageImg(2),...
    img(:,:,3)-averageImg(3));

img = img(:, :, [3, 2, 1]); % convert from RGB to BGR
img = permute(img, [2, 1, 3]); % permute width and height

model = 'VGG_FACE_deploy.prototxt';
weights = 'VGG_FACE.caffemodel';
caffe.set_mode_cpu();
net = caffe.Net(model, weights, 'test'); % create net and load weights

tic
res = net.forward({img});
prob = res{1};
toc
caffe_ft = net.blobs('fc7').get_data();
