/* Read a single facial image and recognize it.
Xiang Xiang (eglxiang@gmail.com), March 2016, MIT license.
*/

#define CPU_ONLY
//#define SIN_INPUT
#define CORR_METRIC
#include "Classifier.h"
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>

using namespace caffe;
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
#ifdef SIN_INPUT
    // load image
    if (argc != 4)
    {
        cerr<<"Run a simple test sample using pretrained VGG face model. " << endl
            << "Usage: " << argv[0]
            << "deploy.prototxt network.caffemodel image" << endl;
            // downloaded from http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz
        return -1;
    }
    string model_file = argv[1];
    string trained_file = argv[2];
    Classifier classifier(model_file, trained_file);
    string file = argv[3];

    cout << "------------ Feature extraction for "<< file<< "------------" << endl;

    Mat img = imread(file, -1);
    if (! img.data) {
        cout<<"Could not open or find the image"<<endl;
        return -1;
    }
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", img);
    int height = img.rows;
    int width = img.cols;

    // subtract the average face
    Mat avg(height,width,CV_8UC3,Scalar(93.5940,104.7624,129.1863));
    Mat image = img - avg;
    namedWindow("Display subtracted face", WINDOW_AUTOSIZE);
    imshow("Display subtracted face", image);

    // foward in the network
    vector<float> output = classifier.Predict(image);

    int count = 0;
   /* cout<<"Prediction:"<<endl;
    for (vector<float>::const_iterator i = output.begin(); i != output.end(); ++i) {
        count = count + 1;
        cout << *i << ' ';
    }
    cout << endl << count << endl;
    cout<<"Prediction done!"<<endl;*/
    cvWaitKey(0);

    return 0;
#else
    if (argc != 5) {
        cerr << "Run a simple test sample using pre-trained VGG face model. " << endl
        << "Usage: " << argv[0]
        << "deploy.prototxt network.caffemodel imageA imageB" << endl;
        return -1;
    }
    string model_file = argv[1];
    string trained_file = argv[2];
    Classifier classifier(model_file, trained_file);
    string datapath = "/home/eglxiang/databases/FaceRecognition/YouTubeFaces/aligned_images_DB/";
    string fileA = argv[3];
    string fileApath = datapath + fileA;
    string fileB = argv[4];
    string fileBpath = datapath + fileB;
    cerr << "--------------------- Feature extraction for "<<fileA<<"-----------"<<endl;
    Mat imgA = imread(fileApath, -1);
    if (! imgA.data) {
        cout << "Could not open or find the image " << fileA << endl;
    }
    namedWindow("Display window 1", WINDOW_AUTOSIZE);
    imshow("Display window 1", imgA);
    int heightA = imgA.rows;
    int widthA = imgA.cols;

    Mat imgB = imread(fileBpath, -1);
    if (! imgB.data) {
        cout << "Could not open or find the image " << fileB << endl;
    }
    namedWindow("Display window 2", WINDOW_AUTOSIZE);
    imshow("Display window 2", imgB);
    int heightB = imgB.rows;
    int widthB = imgB.cols;

    // subtract the average face
    Mat avgA(heightA,widthA,CV_8UC3,Scalar(93.5940,104.7624,129.1863));
    Mat imageA = imgA - avgA;
    Mat avgB(heightB,widthB,CV_8UC3,Scalar(93.5940,104.7624,129.1863));
    Mat imageB = imgB - avgB;

    // foward in the network
    vector<float> outputA = classifier.Predict(imageA);
    vector<float> outputB = classifier.Predict(imageB);

    /*for (vector<float>::const_iterator i = outputA.begin(); i != outputA.end(); ++i) {
        cout << *i << ' ';
    }
    cout << endl << endl << endl;
    for (vector<float>::const_iterator j = outputB.begin(); j != outputB.end(); ++j) {
        cout << *j << ' ';
    }*/

    #ifdef CORR_METRIC
    // compute cosine similarity
    float in_prod = 0;
    for (int i=0; i<outputA.size(); i++)
        in_prod += outputA[i]*outputB[i];
    double sim = in_prod/(norm(outputA,NORM_L2)*norm(outputB,NORM_L2));
    cout << endl << "Similarity: "<< sim*100.0 << "%"<<endl;
    #else
    float sub = 0;
    float sub_norm_sq = 0;
    for (int i=0; i<outputA.size(); i++)
        sub = outputA[i] - outputB[i];
        sub_norm_sq += sub*sub;
    double sub_norm = sqrt(sub_norm_sq);
    cout << endl << "Difference: "<< sub_norm*100.0 << "%"<< endl;
    #endif // CORR_METRIC

    cvWaitKey(0);

#endif // TWO_INPUT
}
