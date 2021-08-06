#ifndef CV_PART_BOATDETECTOR_H
#define CV_PART_BOATDETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

using namespace std;
using namespace cv;

struct Boat {
    Point pt1;
    Point pt2;
    String pred_cl;
    float pred_score;
    float iou;
};

struct TrueBoat {
    Point pt1;
    Point pt2;
};

class BoatDetector {
public:
    BoatDetector(String project_path, String annotation_path, String model_path, float confidence);
    ~BoatDetector();
    float computeImage(String image_path);
    float computeImage(String image_path, bool showImage);
    float computeAll(String folder_path);
    float computeAll(String folder_path, bool showImages);
private:
    std::vector<cv::String> imagesPaths;
    float intersection_over_union(Boat boat, TrueBoat true_boat);
    String project_path;
    String annotation_path;
    String model_path;
    float confidence;
    PyObject *pInit, *pName, *pModule, *pDict, *pFunc, *pValue, *pReturn, *pArgs;
};

#endif //CV_PART_BOATDETECTOR_H