# Packages
# Pillow (PIL) (pip install pillow==8.2.0)
# torchvision
# pycocotools

# #Imports and Drive mount
import os
import numpy as np
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import json
import glob

import torchvision

from engine import train_one_epoch, evaluate
import utils
import transforms as T


class BoatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.train = True
        self.transforms = self.get_transform()

        # Load the annotations from the COCO json file
        self.annotations_json = json.load(open(f'{self.dataset_path}kaggle_annotations.json'))

        self.images_name = []
        for filename in glob.glob(f'{self.dataset_path}*.jpg'):
            self.images_name.append(filename.split('/')[-1])
        

    def __getitem__(self, idx):

        # Initialize the output target
        target = {}

        # Get details of the image with index idx
        image_name = self.images_name[idx] # Image name
        image_path = os.path.join(self.dataset_path, image_name) # Image path
        image = Image.open(image_path).convert("RGB") # Image file
        image_json = list(filter(lambda image: image['file_name'] == image_name, self.annotations_json['images'])) # Image json from the annotations

        ###print(f'Image name: {image_name}')
        # Check if the image is annotated
        if len(image_json) > 0:

            image_json = image_json[0]

            # Get the annotations relevant to the image
            annotations = list(filter(lambda annotation: annotation['image_id'] == image_json['id'], self.annotations_json['annotations']))
            annotations_num = len(annotations)
            ###print(f'    Annotation: {annotations}')
    
            # Calculate bounding boxes (converting from (x1, y1, width, height) to (x1, y1, x2, y2))
            boxes = []
            for annotation in annotations:
                box = annotation['bbox']
                boxes.append([box[0], box[1], box[0]+box[2], box[1]+box[3]])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            # Put the labels to 1 (the only class we have). 0 is the background
            labels = torch.ones((annotations_num,), dtype=torch.int64)

            # Put the image_id equal to the index we provide to the Dataset
            image_id = torch.tensor([idx])

            # Calculate the area of the bounding box
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            # No crowd in our dataset
            iscrowd = torch.zeros((annotations_num,), dtype=torch.int64)
        
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
        else:
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'area': torch.zeros(0, dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.int64)
            }
            ###print('    No annotation')

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images_name)

    def get_transform(self):
        transforms = []

        # Converts the PIL image into a PyTorch Tensor
        transforms.append(T.ToTensor())

        if self.train == True:
            # Flip horizontally and randomly during training
            transforms.append(T.RandomHorizontalFlip(0.5))

        return T.Compose(transforms)

    def set_val(self):
        self.train = False
        self.transforms = self.get_transform()



if __name__ == '__main__':
    # Create the entire dataset
    dataset = BoatDataset('/Users/fabio/Documents/Università/Computer Vision/Progetto/BoatDetection/datasets/kaggle_dataset/')

    # Calculate the number of train and validation images
    train_percentage = 0.8
    train_images_num = round(len(dataset) * 0.8)
    val_images_num = len(dataset) - train_images_num
    print(f'Train dataset size = {train_images_num}, Val dataset size = {val_images_num}')

    # Split the original dataset into the train dataset and the validation dataset
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_images_num, val_images_num], generator=torch.Generator().manual_seed(42))

    # Set the train mode to False
    val_dataset.dataset.set_val()

    # Initialize training and validation DataLoaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)


    # #Model architecture

    # In[25]:





    def build_model(num_classes):
        # Load an instance of a pre-trained model (Faster-RCNN)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model


    # #Initialize model and optimizer

    # Choose which device to use
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Out model has two classes (background = 0 and boat = 1)
    num_classes = 2

    # Get the model using the helper function
    model = build_model(num_classes)

    # Move the model to the device
    model.to(device)

    # Construct the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler which decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


    # #Training the model

    # Number of epochs
    num_epochs = 10

    for epoch in range(num_epochs):

        # Train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=1)

        # Update the learning rate
        lr_scheduler.step()

        # Evaluate on the val dataset
        evaluate(model, val_dataloader, device=device)