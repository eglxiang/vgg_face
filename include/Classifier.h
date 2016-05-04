#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace caffe;
using namespace cv;
using namespace std;


class Classifier {
    public:
        Classifier(const string& model_file,const string& trained_file);
        vector<float> Predict(const Mat& img);
    private:
        void WrapInputLayer(vector<Mat>* input_channels);
        void Preprocess(const Mat& img, vector<Mat>* input_channels);
    private:
        shared_ptr<Net<float> > net_;
        Size input_geometry_;
        int num_channels_;
};


#endif // CLASSIFIER_H
