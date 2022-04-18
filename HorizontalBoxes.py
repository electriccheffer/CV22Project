'''

The goal of this program is to transform a bounding box rotated by theta
radians into a horizontal bounding box which contains the original.

    [x]Change Annotations
    [x]Visualize Annotations
    []Load into Keras
    []Preprocessing
    []Raw Training
    []Pretrained model
    []Visualize predictions

'''
import json as js
import math as mth
import os


# function to perform our box transformation takes in a list of the mvtec_box annots
# returns a new four parameter list to be encoded onto our new json file
def new_box(old_boxes):
    w1 = old_boxes[3] * mth.cos(mth.pi - (mth.pi / 2) - old_boxes[4])
    w2 = old_boxes[2] * mth.cos(old_boxes[4])
    w = mth.fabs(w1) + mth.fabs(w2)
    l1 = old_boxes[3] * mth.cos(old_boxes[4])
    l2 = old_boxes[2] * mth.cos(mth.pi - mth.pi / 2 - old_boxes[4])
    l = mth.fabs(l1) + mth.fabs(l2)
    new_boxes = [old_boxes[1] - w/2, old_boxes[0] - l/2, w, l, 0]
    return new_boxes


def create_new_json(path):

    new_json = js.load(open("./jsonFiles/" + path, 'r'))

    # Replace annotation for bbox in each annotation for the new json file
    for annot in new_json["annotations"]:
        new_bbox = new_box(annot["bbox"])
        annot["bbox"] = new_bbox
        annot["area"] = new_bbox[2] * new_bbox[3]
    new_json_file = open("./new_JSON/new_" + path,'w')
    js.dump(new_json, new_json_file)


# Change Annotations
'''
Changes the annotations given to us by mv_tec into a COCO formated sheet.  
'''
for file_path in os.listdir("./jsonFiles"):
    create_new_json(file_path)
