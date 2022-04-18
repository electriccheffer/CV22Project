
###############################################
###############################################
###############################################
###############################################
# Visualize Annotations
'''
Randomly selects ten images and draws their new bounding boxes.
Run this program until you feel satisfied that the transformation
was correct.
'''
# import required libraries
import json as js
import matplotlib.pyplot as mplt
import random as rand
from PIL import Image
from matplotlib import patches


# Generate a list of image file names using random numbers the function returns
# the list of file objects and a list of image file names
def random_images(ran_num):

    #file paths from random_numbers
    image_name_list = []
    for number in ran_num:
        string_number = str(number)
        if 10 - number > 0:
            image_name_list.append("screws_00" + string_number + ".png")
        elif 100 - number > 0:
            image_name_list.append("screws_0" + string_number + ".png")
        elif 1000 - number > 0:
            image_name_list.append("screws_" + string_number + ".png")

    return image_name_list


# get and return the correct annotations for the given image
# images may be in either training, test, or validation sets for
# annotations.  Therefore we must check each annotation file
# to get the correct annotations.
def get_annotations(image_names):

    # create json objects for each of the annotation files
    test_annots = js.load(open("new_JSON/new_mvtec_screws_test.json", 'r'))
    train_annots = js.load(open("new_JSON/new_mvtec_screws_train.json", 'r'))
    val_annots = js.load(open("new_JSON/new_mvtec_screws_val.json", 'r'))

    annot_list = []

    # iterate over the list of random numbers
    for image_name in image_names:

        # check which set the image is in
        for image_annots in test_annots["images"]:
            if image_annots["file_name"] == image_name:
                annot_list.append("test")
                annot_list.append(image_annots["id"])
                break

        for image_annots in train_annots["images"]:
            if image_annots["file_name"] == image_name:
                annot_list.append("train")
                annot_list.append(image_annots["id"])
                break

        for image_annots in val_annots["images"]:
            if image_annots["file_name"] == image_name:
                annot_list.append("val")
                annot_list.append(image_annots["id"])
                break

    return annot_list

# Finds the annotations for the imagelist given image list contains the
# file paths for the images are contained within the image_list the annotation
# list contains the id number of the image.
def find_annot_for_image(image_list, annotation_list):


    test_annot_file = js.load(open("new_JSON/new_mvtec_screws_test.json", 'r'))
    train_annot_file = js.load(open("new_JSON/new_mvtec_screws_train.json", 'r'))
    val_annot_file = js.load(open("new_JSON/new_mvtec_screws_val.json", 'r'))
    index = 0
    image_list_index = 0

    while index <= len(annotation_list):

        sheet = annotation_list[index]
        index += 1
        id = annotation_list[index]

        if sheet == "test":
            for annot in test_annot_file["annotations"]:
                if annot["image_id"] == id:
                    bbox = annot["bbox"]
                    cat_id = annot["category_id"]
                    draw_image(image_list[image_list_index], bbox)
                    index += 1
                    image_list_index += 1
                    break

        elif sheet == "train":
            for annot in train_annot_file["annotations"]:
                if annot["image_id"] == id:
                    bbox = annot["bbox"]
                    cat_id = annot["category_id"]
                    draw_image(image_list[image_list_index], bbox)
                    image_list_index += 1
                    index += 1
                    break

        elif sheet == "val":
            for annot in val_annot_file["annotations"]:
                if annot["image_id"] == id:
                    bbox = annot["bbox"]
                    cat_id = annot["category_id"]
                    draw_image(image_list[image_list_index], bbox)
                    index += 1
                    image_list_index += 1
                    break

        if index == len(annotation_list):
            break

# takes in the image file name and its bbox, places the image in matplot
# draws the images and the bounding boxes.
def draw_image(image, bbox):

    # create an image
    im = Image.open("./train/" + image)

    # put the image into matplot
    fig, axis = mplt.subplots()
    axis.imshow(im)

    # create the rectangle
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 0, linewidth=1, edgecolor='r', facecolor='none')
    axis.add_patch(rect)
    mplt.show()


# generate ten random numbers
random_numbers = []
for i in range(10):
    random_numbers.append((rand.randrange(1, 384)))

# select ten random image files
image_nm_list = random_images(random_numbers)

# select the annotations for the images
annotations = get_annotations(image_nm_list)

# Draw the images on the picture
find_annot_for_image(image_nm_list, annotations)
