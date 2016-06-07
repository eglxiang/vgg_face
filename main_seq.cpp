/* Process all images in the directory specified in the argument.
Xiang Xiang (eglxiang@gmail.com), May 2016, MIT license.
Main functionality: compute VGG_Face features for each image.
First you need to create the saving directory yourself.
It writes the feature vector of each face image into a txt file.
*/

#define CPU_ONLY
#define CORR_METRIC
#include "Classifier.h"
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <dirent.h>
#include <string.h>
//#include <boost/python.hpp>

using namespace caffe;
using namespace cv;
using namespace std;
//using namespace boost::python

int main(int argc, char** argv)
{
    // load image
    if (argc != 4)
    {
        cerr<<"Run a simple test sample using pretrained VGG face model. " << endl
            << "Usage: " << argv[0]
            << "deploy.prototxt network.caffemodel folderpath" << endl;
        return -1;
    }
    string model_file = argv[1];
    string trained_file = argv[2];
    Classifier classifier(model_file, trained_file);
    string foldername = argv[3];
    string rootpath = "/mnt/localsata/selected_faces/";
    string folderpath = rootpath + foldername;

    // list files in the folder
    DIR *dir;
    dir = opendir(folderpath.c_str());
    cout << dir << endl;
    string imgName;
    struct dirent *ent;

    if (dir != NULL) {
        while ( (ent = readdir(dir)) != NULL ) {
            imgName = ent->d_name;
            if (imgName.compare(".")!=0 && imgName.compare("..")!=0)
            {
                string aux;
                aux.append(folderpath);
                aux.append(imgName);
                cout << aux << endl;
                Mat img = imread(aux, -1);
                if (! img.data) {
                    cout << "Could not open or find the image" << endl;
                    return -1;
                }
                int height = img.rows;
                int width = img.cols;

                // subtract the average face
                Mat avg(height, width, CV_8UC3, Scalar(93.5940,104.7624,129.1863));
                // Mat mean_img(height, width, CV_8UC3, Scalar(90,100,120)); //also works
                Mat image = img - avg;
                vector<float> output = classifier.Predict(image);

                //for (vector<float>::const_iterator i = output.begin(); i != output.end(); ++i) {
                //    cout << *i << ' ';
                //}
            }
        }
    }

    return 0;
}
