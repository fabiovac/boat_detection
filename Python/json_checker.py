import json

f = open('/Users/fabio/Documents/Università/Computer Vision/Progetto/BoatDetection/CV_part/cmake-build-debug/datasets/kaggle_annotations.json')
json_file = json.load(f)

annotations = json_file['annotations']

allright = True
for annotation in annotations:
    bbox = annotation['bbox']
    if bbox[2] <= 0 or bbox[3] <= 0:
        print(f'Problem with image id:{annotation["image_id"]} - Bounding box:{bbox}')
        allright = False

if allright:
    print('The json is correctly formatted')


    