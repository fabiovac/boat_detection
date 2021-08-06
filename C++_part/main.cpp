#include "BoatDetector.h"
#include "PreProcessor.h"


String getParamPath(CommandLineParser parser, String paramName, String project_path, bool endSlash) {
    //Get the folder path and add / in front if not present
    String param_name = parser.get<String>(paramName);
    if (param_name[0] != '/') param_name = "/" + param_name;
    String param_path = project_path + param_name;
    if (endSlash && !param_path.empty() && param_path.back() != '/')
        param_path += '/';
    return param_path;
}

String getParamPath(CommandLineParser parser, String paramName, String project_path) {
    return getParamPath(parser, paramName, project_path, true);
}

int main(int argc, char *argv[]) {

    const String keys =
            "{@mode            |                                                 | prepro, eval or evalAll                                  }"
            "{image            |                                                 | Image on which to evaluate                               }"
            "{folder           |                                                 | Folder where to find images on which to use evaluate     }"
            "{output           |                                                 | Folder where to put the output of the pre-processing     }"
            "{annotation       | datasets/kaggle_annotations.json                | Annotations json file name/path                          }"
            "{annotationoutput | datasets/kaggle_annotations_prepro.json         | Annotations json file name/path for the modified json    }"
            "{model            | models/model_v2.pth                             | Model name/path on which to perform the evaluation       }"
            "{dataset          | datasets/kaggle_dataset                         | Dataset name/path on which to perform the pre-processing }"
    ;
    CommandLineParser parser(argc, argv, keys);

    //Get the project path
    String executable_path = argv[0];
    String project_path = executable_path.substr(0, executable_path.find_last_of("\\/"));

    //Get the program mode
    String mode = parser.get<String>("@mode");
    if (mode == "eval") {
        cout << "Evaluation mode (eval)" << endl;

        //Get the parameters from the input
        String image_path = getParamPath(parser, "image", project_path, false);
        String annotation_path = getParamPath(parser, "annotation", project_path, false);
        String model_path = getParamPath(parser, "model", project_path, false);

        //Initialize the BoatDetector object and run the proper method on it
        BoatDetector boatDetector(project_path, annotation_path, model_path, 0.95);
        boatDetector.computeImage(image_path, true);

    } else if (mode == "evalAll") {
        cout << "Evaluation test set mode (evalAll)" << endl;

        //Get the parameters from the input
        String annotation_path = getParamPath(parser, "annotation", project_path, false);
        String folder_path = getParamPath(parser, "folder", project_path);
        String model_path = getParamPath(parser, "model", project_path, false);

        //Initialize the BoatDetector object and run the proper method on it
        BoatDetector boatDetector(project_path, annotation_path, model_path, 0.95);
        boatDetector.computeAll(folder_path, true);

    } else if (mode == "prepro") {
        cout << "Dataset pre-processing mode (prepro)" << endl;

        //Get the parameters from the input
        String annotation_path = getParamPath(parser, "annotation", project_path, false);
        String annotation_output_path = getParamPath(parser, "annotationoutput", project_path, false);
        String folder_path = getParamPath(parser, "folder", project_path);
        String output_path = getParamPath(parser, "output", project_path);

        //Initialize the PreProcessor object and run the compute method on it
        PreProcessor preProcessor(annotation_path, annotation_output_path, folder_path, output_path);
        preProcessor.compute();
    } else {
        cout << "Mode not supported (prepro, eval, evalAll only)";
    }

    return 0;
}