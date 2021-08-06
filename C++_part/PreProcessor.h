#ifndef CV_PART_PREPROCESSOR_H
#define CV_PART_PREPROCESSOR_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include "include/nlohmann/json.hpp"

using json = nlohmann::json;
using namespace std;
using namespace cv;

class PreProcessor {
public:
    PreProcessor(String annotation_path, String annotation_output_path, String folder_path, String output_path);
    ~PreProcessor();
    void compute();
private:
    String annotation_path;
    String annotation_output_path;
    String folder_path;
    String output_path;
    json images;
    json annotations;
    vector<String> imagesPaths;
    map<String, tuple<int, int, int>> imageMap;
    vector<int> H_min_val;
    vector<int> H_max_val;
    vector<int> S_val;
    vector<int> V_val;
    int minScore;
    int cluster_size;

    bool isSelected(unsigned char H, unsigned char S, unsigned char V);
};

#endif //CV_PART_PREPROCESSOR_H