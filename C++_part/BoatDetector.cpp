#include "BoatDetector.h"


BoatDetector::BoatDetector(String project_path, String annotation_path, String model_path, float confidence) {
    this->project_path = project_path;
    this->annotation_path = annotation_path;
    this->model_path = model_path;
    this->confidence = confidence;

    //Initialize the Python Interpreter
    Py_Initialize();

    //Get and print the Python version it is using
    String pythonVersion = Py_GetVersion();
    cout << "Python: " << pythonVersion.substr(0, pythonVersion.find(" (")) << endl;

    cout << "Project path: " << project_path << endl;
    cout << "Annotation path: " << annotation_path << endl;
    cout << "Model path: " << model_path << endl;

    //Add the directory to find the python script in the same path as the executable
    //Taken from: https://stackoverflow.com/questions/21279913/python-embedding-pyimport-import-not-from-the-current-directory
    String pythonDirectory = this->project_path + "/python";
    PyObject* sysPath = PySys_GetObject((char*)"path");
    PyObject* programName = PyUnicode_FromString(pythonDirectory.c_str());
    PyList_Append(sysPath, programName);
    Py_DECREF(programName);

    //Load the module object
    pName = PyUnicode_FromString("python_code");
    pModule = PyImport_Import(pName);

    //pDict is a borrowed reference
    pDict = PyModule_GetDict(pModule);

    //pFunc is also a borrowed reference
    pName = PyUnicode_FromString("init");
    pInit = PyDict_GetItem(pDict, pName);

    //pFunc is also a borrowed reference
    pName = PyUnicode_FromString("main");
    pFunc = PyDict_GetItem(pDict, pName);

    if (PyCallable_Check(pInit)) {
        pArgs = PyTuple_New(3);

        pValue = PyUnicode_FromString(this->model_path.c_str());
        PyTuple_SetItem(pArgs, 0, pValue);

        pValue = PyUnicode_FromString(this->annotation_path.c_str());
        PyTuple_SetItem(pArgs, 1, pValue);

        pValue = PyFloat_FromDouble(this->confidence);
        PyTuple_SetItem(pArgs, 2, pValue);

        PyObject_CallObject(pInit, pArgs);
    } else {
        PyErr_Print();
        cout << "BoatDetector not initialized correctly" << endl;
    }
}

BoatDetector::~BoatDetector() {
    //Clean up
    Py_DECREF(pModule);
    Py_DECREF(pName);

    //Finish the Python Interpreter
    Py_Finalize();
}

float BoatDetector::intersection_over_union(Boat boat, TrueBoat true_boat) {
    //Calculate the coordinates of the intersection rectangle
    float r_x1 = max(boat.pt1.x, true_boat.pt1.x);
    float r_y1 = max(boat.pt1.y, true_boat.pt1.y);
    float r_x2 = min(boat.pt2.x, true_boat.pt2.x);
    float r_y2 = min(boat.pt2.y, true_boat.pt2.y);

    //Compute the area of both the prediction and ground-truth rectangles
    float bb_1_area = (boat.pt2.x - boat.pt1.x + 1) * (boat.pt2.y - boat.pt1.y + 1);
    float bb_2_area = (true_boat.pt2.x - true_boat.pt1.x + 1) * (true_boat.pt2.y - true_boat.pt1.y + 1);

    //Compute the area of intersection rectangle
    float inter_area = max(0.0f, r_x2 - r_x1 + 1) * max(0.0f, r_y2 - r_y1 + 1);

    //Compute the intersection over union
    float iou = inter_area / (bb_1_area + bb_2_area - inter_area);

    return iou;
}

float BoatDetector::computeImage(String image_path, bool showImage) {
    cout << "Image path: " << image_path << endl << endl;

    vector<Boat> boats;
    vector<TrueBoat> true_boats;
    float iou_tot = 0.0;

    //Read the image
    Mat img = imread(image_path, IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    //Convert the color space
    cvtColor(img, img, COLOR_BGR2RGB);

    if (PyCallable_Check(pFunc)) {
        pArgs = PyTuple_New(1);

        pValue = PyUnicode_FromString(image_path.c_str());
        PyTuple_SetItem(pArgs, 0, pValue);

        pReturn = PyObject_CallObject(pFunc, pArgs);

        if (PyTuple_Check(pReturn)) {
            //Get the predicted class labels
            PyObject *pred_cls = PyTuple_GetItem(pReturn, 0);
            if (PyList_Check(pred_cls)) {
                for (Py_ssize_t i = 0; i < PyList_Size(pred_cls); i++) {
                    //Create a new boat and insert into the list
                    struct Boat boat;
                    boats.push_back(boat);

                    PyObject *pred_cl = PyList_GetItem(pred_cls, i);
                    const char *pred_cls_string = PyUnicode_AsUTF8(pred_cl);

                    boats[i].pred_cl = pred_cls_string;
                }
            } else {
                cout << "The first value in the tuple is NOT a list" << endl;
            }

            //Get the predictions scores
            PyObject *pred_scores = PyTuple_GetItem(pReturn, 1);
            if (PyList_Check(pred_scores)) {
                for (Py_ssize_t i = 0; i < PyList_Size(pred_scores); i++) {
                    PyObject *pred_score = PyList_GetItem(pred_scores, i);
                    float pred_score_float = PyFloat_AsDouble(pred_score);

                    boats[i].pred_score = pred_score_float;
                }
            } else {
                cout << "The second value in the tuple is NOT a list" << endl;
            }

            //Get the bounding boxes
            PyObject *boxes = PyTuple_GetItem(pReturn, 2);
            if (PyList_Check(boxes)) {
                for (Py_ssize_t i = 0; i < PyList_Size(boxes); i++) {
                    PyObject *box = PyList_GetItem(boxes, i);
                    if (PyList_Check(box)) {
                        PyObject *point_1 = PyList_GetItem(box, 0);
                        PyObject *point_2 = PyList_GetItem(box, 1);

                        if (PyTuple_Check(point_1) && PyTuple_Check(point_2)) {
                            //Get the x1 and y1 from the point 1
                            PyObject *x_1 = PyTuple_GetItem(point_1, 0);
                            PyObject *y_1 = PyTuple_GetItem(point_1, 1);
                            float x_1_float = PyFloat_AsDouble(x_1);
                            float y_1_float = PyFloat_AsDouble(y_1);

                            //Get the x2 and y2 from the point 2
                            PyObject *x_2 = PyTuple_GetItem(point_2, 0);
                            PyObject *y_2 = PyTuple_GetItem(point_2, 1);
                            float x_2_float = PyFloat_AsDouble(x_2);
                            float y_2_float = PyFloat_AsDouble(y_2);

                            //Compute the points for the rectangle
                            Point pt1(x_1_float, y_1_float);
                            Point pt2(x_2_float, y_2_float);

                            boats[i].pt1 = pt1;
                            boats[i].pt2 = pt2;
                        } else {
                            cout << "One of the points is NOT a tuple" << endl;
                        }
                    } else {
                        cout << "The box is NOT a list" << endl;
                    }
                }
            } else {
                cout << "The third value in the tuple is NOT a list" << endl;
            }

            //Get the ground-truth bounding boxes
            PyObject *gt_boxes = PyTuple_GetItem(pReturn, 3);
            if (PyList_Check(gt_boxes)) {
                for (Py_ssize_t i = 0; i < PyList_Size(gt_boxes); i++) {
                    PyObject *gt_box = PyList_GetItem(gt_boxes, i);
                    if (PyList_Check(gt_box)) {
                        PyObject *point_1 = PyList_GetItem(gt_box, 0);
                        PyObject *point_2 = PyList_GetItem(gt_box, 1);

                        if (PyTuple_Check(point_1) && PyTuple_Check(point_2)) {
                            //Get the x1 and y1 from the point 1
                            PyObject *x_1 = PyTuple_GetItem(point_1, 0);
                            PyObject *y_1 = PyTuple_GetItem(point_1, 1);
                            float x_1_float = PyFloat_AsDouble(x_1);
                            float y_1_float = PyFloat_AsDouble(y_1);

                            //Get the x2 and y2 from the point 2
                            PyObject *x_2 = PyTuple_GetItem(point_2, 0);
                            PyObject *y_2 = PyTuple_GetItem(point_2, 1);
                            float x_2_float = PyFloat_AsDouble(x_2);
                            float y_2_float = PyFloat_AsDouble(y_2);

                            //Compute the points for the rectangle
                            Point pt1(x_1_float, y_1_float);
                            Point pt2(x_2_float, y_2_float);

                            //Create a TrueBoat and assign the points of the bounding box returned by python
                            struct TrueBoat true_boat;
                            true_boat.pt1 = pt1;
                            true_boat.pt2 = pt2;

                            //Add to the list
                            true_boats.push_back(true_boat);
                        } else {
                            cout << "One of the points is NOT a tuple" << endl;
                        }
                    } else {
                        cout << "The box is NOT a list" << endl;
                    }
                }
            } else {
                cout << "The fourth value in the tuple is NOT a list" << endl;
            }
        } else {
            cout << "The return is NOT a tuple" << endl;
        }

        //Calculate the IoU for the boxes found
        for (int i = 0; i < boats.size(); ++i) {
            float iou_max = 0;
            for (int j = 0; j < true_boats.size(); ++j) {
                //Calculate the IoU
                float iou = intersection_over_union(boats[i], true_boats[j]);
                iou_max = max(iou, iou_max);
            }
            boats[i].iou = iou_max;
            iou_tot = iou_tot + iou_max;
        }
        if (boats.size() > 0)
            iou_tot = iou_tot / boats.size();

        //Print and show the image with the bounding boxes nly if showImage is set to true
        if (showImage) {
            cout << "Image: " << image_path << endl;

            //Convert the color space
            cvtColor(img, img, COLOR_RGB2BGR);

            //Print to the console the results and draw the bounding boxes and the other infos on a new window
            for (int i = 0; i < boats.size(); ++i) {
                Boat boat = boats[i];
                cout << "Bounding box: (" << boats[i].pt1.x << ", " << boats[i].pt1.y << ") -> (" << boats[i].pt2.x << ", " << boats[i].pt2.y << ") - ";
                cout << "Score: " << boat.pred_score << " - ";
                cout << "Class: " << boat.pred_cl << " - ";
                cout << "IoU: " << boat.iou << endl;

                //Draw the rectangle
                rectangle(img, boat.pt1, boat.pt2, Scalar(255, 0, 0));
                String label = boat.pred_cl + " " + to_string(i) + " - Score: " + to_string(boat.pred_score) + " - IoU: " + to_string(boat.iou);
                Point label_point = Point(boat.pt1.x, boat.pt1.y - 5);
                putText(img, label, label_point, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
            }

            //Draw the bounding boxes of the ground-truth
            for (int i = 0; i < true_boats.size(); ++i) {
                TrueBoat true_boat = true_boats[i];

                //Draw the rectangle
                rectangle(img, true_boat.pt1, true_boat.pt2, Scalar(0, 255, 0));
            }

            //Print the general IoU
            cout << "General IoU: " << iou_tot << endl << endl;

            imwrite("/Users/fabio/Documents/Università/Computer Vision/Progetto/BoatDetection/CV_part/cmake-build-debug/datasets/output/image.jpg", img);
            imshow("Result", img);
            waitKey(0);
        }
    } else {
        PyErr_Print();
    }

    return iou_tot;
}

float BoatDetector::computeImage(String image_path) {
    return computeImage(image_path, true);
}

float BoatDetector::computeAll(String folder_path, bool showImages) {
    cout << "Folder path: " << folder_path << endl << endl;

    //Get the images from the folder
    cv::glob(folder_path, imagesPaths);

    float iou_tot = 0;
    for (int i = 0; i < imagesPaths.size(); ++i) {
        float iou_img = computeImage(imagesPaths[i], showImages);
        iou_tot = iou_tot + iou_img;
    }
    iou_tot = iou_tot / imagesPaths.size();

    cout << endl << "The average IoU of the test set is: " << iou_tot << endl;

    return iou_tot;
}

float BoatDetector::computeAll(String folder_path) {
    return computeAll(folder_path, false);
}