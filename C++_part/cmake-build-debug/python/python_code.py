import torch
import torchvision
import torchvision.transforms as T
import json
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

import warnings
warnings.filterwarnings("ignore")


model_path = ""
annotation_path = ""
annotations_json = None
confidence = 0.95
device = None
model = None
class_names = None
import sys

"""#Model architecture"""
def build_model(num_classes):
    # Load an instance of a pre-trained model (Faster-RCNN)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


"""#Get the groud-truth bounding boxes"""
def gt_bb(image_name):

    # Extract the json part where the image is defined
    image_json = list(filter(lambda image: image['file_name'] == image_name, annotations_json['images']))

    boxes = []
    if len(image_json) > 0:
        image_json = image_json[0]

        # Extract the json part where the annotations related to the image are defined
        annotations = list(filter(lambda annotation: annotation['image_id'] == image_json['id'], annotations_json['annotations']))

        # Calculate bounding boxes (converting from (x1, y1, width, height) to (x1, y1, x2, y2))
        for annotation in annotations:
            box = annotation['bbox']
            boxes.append([box[0], box[1], box[0]+box[2], box[1]+box[3]])

    # Return the bounding boxes related to the image
    return boxes


def get_prediction(image_path):

    # Open the image from the image_path
    image = Image.open(image_path)

    # Prepare the image to be fed to the NN
    transform = T.Compose([T.ToTensor()])
    image = transform(image).to(device)

    # Calculate the prediction with the NN
    pred = model([image])

    pred_class = [class_names[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())

    pred_t = [pred_score.index(x) for x in pred_score if x > confidence]
    if len(pred_t) > 0:
        pred_t = pred_t[-1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        pred_score = pred_score[:pred_t+1]
    else:
        pred_boxes = []
        pred_class = []
        pred_score = []

    return pred_boxes, pred_class, pred_score


def init(model_path_t, annotation_path_t, confidence_t):

    global model_path, annotation_path, annotations_json, confidence, device, model, class_names

    model_path = model_path_t
    annotation_path = annotation_path_t
    confidence = confidence_t

    # Choose which device to use
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Out model has two classes (background = 0 and boat = 1)
    num_classes = 2

    # Get the model using the helper function
    model = build_model(num_classes)

    try:
        # Load the model from the file
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
    except FileNotFoundError:
        print('Model file not found')

    # Define the class names
    class_names = ['__background__', 'boat']

    # Load the annotation json
    try:
        annotations_file = open(annotation_path)
        annotations_json = json.load(annotations_file)
    except FileNotFoundError:
        print('Annotation file not found')


def main(image_path):
    boxes, pred_cls, pred_score = get_prediction(image_path)

    # Calculate the ground-truth bounding boxes
    gt_boxes = gt_bb(image_path.split('/')[-1])
    gt_boxes = [[(gt_box[0], gt_box[1]), (gt_box[2], gt_box[3])] for gt_box in gt_boxes]

    return (pred_cls, pred_score, boxes, gt_boxes)