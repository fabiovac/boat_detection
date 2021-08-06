#include "PreProcessor.h"

PreProcessor::PreProcessor(String annotation_path, String annotation_output_path, String folder_path,
                           String output_path) {
    this->annotation_path = annotation_path;
    this->annotation_output_path = annotation_output_path;
    this->folder_path = folder_path;
    this->output_path = output_path;

    cout << "Annotation path: " << annotation_path << endl;
    cout << "Annotation output path: " << annotation_output_path << endl;
    cout << "Folder path: " << folder_path << endl;
    cout << "Output folder path: " << output_path << endl << endl;

    //Open the annotation file and read it as a json object
    std::ifstream file(annotation_path);
    file >> this->annotations;

    //Creates a map with images names as key and id as value
    this->images = this->annotations["images"];
    for (auto& el : this->images.items()) {
        json image = el.value();

        //Get image details (file name, id, width, height)
        String fileName = image["file_name"];
        fileName.erase(remove(fileName.begin(), fileName.end(), '\"'),fileName.end());
        int id = image["id"];
        int width = image["width"];
        int height = image["height"];

        //Append details in the Map
        this->imageMap.insert(pair<String, tuple<int, int, int>>(fileName, tuple<int, int, int>(id, width, height)));
    }

    //Get the list of all the images to pre-process
    vector<String> imagesPaths;
    glob(folder_path, this->imagesPaths);

    //Parameters to tune

    //Kaggle
    this->H_min_val = {200, 100};
    this->H_max_val = {225, 195};
    this->S_val = {25, 8};
    this->V_val = {28, 21};

    //MAR
    /*this->H_min_val = {50, 105};
    this->H_max_val = {75, 205};
    this->S_val = {20, 20};
    this->V_val = {45, 85};*/

    this->minScore = 30;
    this->cluster_size = 8;
}

//Kaggle
bool PreProcessor::isSelected(unsigned char H, unsigned char S, unsigned char V) {
    return (H_min_val[0]/360.0*179) < H && H < (H_max_val[0]/360.0*179) && (S_val[0]/100.0*255) <= S && (V_val[0]/100.0*255) <= V || //This is for the sky and the blue water
           (H_min_val[1]/360.0*179) < H && H < (H_max_val[1]/360.0*179) && (S_val[1]/100.0*255) <= S && (V_val[1]/100.0*255) <= V;   //This is for the green water
}

//MAR
/*bool PreProcessor::isSelected(unsigned char H, unsigned char S, unsigned char V) {
    return (H_min_val[0]/360.0*179) < H && H < (H_max_val[0]/360.0*179) && (S_val[0]/100.0*255) > S && (V_val[0]/100.0*255) > V ||
           (H_min_val[1]/360.0*179) < H && H < (H_max_val[1]/360.0*179) && (S_val[1]/100.0*255) > S && (V_val[1]/100.0*255) > V;
}*/

void PreProcessor::compute() {
    //Foreach image to pre-process
    for (int i = 0; i < this->imagesPaths.size(); ++i) {
        //Get the image path, image name and extension
        String imagePath = this->imagesPaths[i];
        int lastSlashPos = imagePath.find_last_of("/");
        String imageName = imagePath.substr(lastSlashPos+1, imagePath.size() - lastSlashPos);
        int pointPos = imageName.find_last_of(".");
        String extension = imageName.substr(pointPos+1, imageName.size() - pointPos);

        //Compute the new image name
        int pointLastPos = imageName.find_last_of ('.');
        String imageNameNoExt = imageName.substr(0, pointLastPos);
        String imageExt = imageName.substr(pointLastPos, imageName.size());
        String newImageName = imageNameNoExt + "-selectiveblur" + imageExt;

        //Proceed only if it is a jpg/jpeg/png file format
        if (extension == "jpg" || extension == "jpeg" || extension == "png") {
            cout << "Pre-processing image: " << imageName << " - ";

            //Search for the image in the image dict
            auto imagePair = this->imageMap.find(imageName);

            //If the image is not found means that this image is not in the annotation file so it won't be used for
            //positive training so we don't need any segmented version
            if (imagePair == imageMap.end()) {
                cout << "No annotation for this image -> Skipped" << endl;
                continue;
            }

            //Get the image id, width and height
            int imageId = get<0>(imagePair->second);
            int width = get<1>(imagePair->second);
            int height = get<2>(imagePair->second);

            //Open the image file
            Mat image = imread(imagePath, IMREAD_COLOR);

            //Save the original image into the new location without any pre-processing
            imwrite(output_path+imageName, image);

            // STARTING PRE-PROCESSING //

            //Convert the image from BGR to HSV
            cvtColor(image, image, COLOR_BGR2HSV);

            //Define some variables for the k-means method
            vector<Point3f> centers;
            Mat labels;

            //Create a data matrix from which k-means takes the points
            Mat data;
            image.convertTo(data, CV_32F);
            data = data.reshape(1, data.total());

            //Definition of the termination criteria and run of the k-means clustering algorithm
            TermCriteria termCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 0.2);
            kmeans(data, cluster_size, labels, termCriteria, 10, KMEANS_RANDOM_CENTERS, centers);

            //Reshape the output with the original image shape
            labels = labels.reshape(1, image.rows);

            //Count the pixel of each segment which are considered as selected (blue ones) and the total number
            vector<int> segmentsCount = vector<int>(cluster_size);
            vector<int> segmentsSelectedCount = vector<int>(cluster_size);
            for (int y = 0; y < labels.rows; ++y) {
                for (int x = 0; x < labels.cols; ++x) {
                    int currentSegmentId = labels.at<int>(y, x);
                    Vec3b point = image.at<Vec3b>(y, x);
                    unsigned char H = point.val[0];
                    unsigned char S = point.val[1];
                    unsigned char V = point.val[2];

                    if (isSelected(H, S, V)) {
                        segmentsSelectedCount[currentSegmentId] += 1;
                    }
                    segmentsCount[currentSegmentId] += 1;
                }
            }

            //Calculate the score for each segment. It is the percentage of selected pixels over the total count
            vector<float> segmentsScore = vector<float>(cluster_size);
            for (int segmentId = 0; segmentId < cluster_size; ++segmentId) {
                segmentsScore[segmentId] = segmentsSelectedCount[segmentId] / (float) segmentsCount[segmentId] * 100;
                cout << segmentsScore[segmentId] << " ";
            }
            cout << endl;

            //Create a mask where white pixels correspond to the area where there are a segment considered to be selected
            //A segment is selected if a has a score at least minScore (a parameter to tune)
            Mat mask = Mat::zeros(image.size(), CV_8UC1);
            for (int segmentId = 0; segmentId < cluster_size; ++segmentId) {
                if (segmentsScore[segmentId] >= minScore) {
                    for (int y = 0; y < image.rows; ++y) {
                        for (int x = 0; x < image.cols; ++x) {
                            int currentSegmentId = labels.at<int>(y, x);
                            if (currentSegmentId == segmentId) {
                                mask.at<unsigned char>(y, x) = 255;
                            }
                        }
                    }
                }
            }

            //Convert the color space from HSV to BGR
            cvtColor(image, image, COLOR_HSV2BGR);

            //Blur the image and save it into a new imageBlurred Mat object
            Mat imageBlurred;
            GaussianBlur(image, imageBlurred, Size(25, 25), 0, 0 );

            /*imshow("Mask", mask);
            waitKey(0);

            imshow("Original Image", image);
            waitKey(0);

            imshow("Gaussian Blur", imageBlurred);
            waitKey(0);*/

            //Blend the blurred part into the original image only where the mask is positive
            for (int y = 0; y < image.rows; ++y) {
                for (int x = 0; x < image.cols; ++x) {
                    if (mask.at<unsigned char>(y, x) == 255) {
                        image.at<Vec3b>(y, x) = imageBlurred.at<Vec3b>(y, x);
                    }
                }
            }

            /*imshow("Result", image);
            waitKey(0);*/

            imwrite(output_path+newImageName, image);

            // END PRE-PROCESSING //

            //=== Save the imagePair in the json file as a new imagePair ===//

            //Create a json object representing the new imagePair
            json newImage;
            newImage["id"] = imageId;
            newImage["file_name"] = newImageName;
            newImage["width"] = width;
            newImage["height"] = height;

            //Append the new imagePair to the json object
            this->images.push_back(newImage);
        }
    }
}

PreProcessor::~PreProcessor() {
    //Update the json
    this->annotations["images"] = this->images;

    //Save the new json file
    std::ofstream fileOutput(this->annotation_output_path);
    fileOutput << std::setw(4) << this->annotations << endl;
}