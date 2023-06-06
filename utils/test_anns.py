import random
import xml.etree.ElementTree as ET
import os
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import ImageDraw, Image, ImageFont
import matplotlib.pyplot as plt

root_dir = "dataset/instruments18/"
dest_dir = root_dir
annotations = glob(root_dir + "*/xml/*.txt")

class_name_to_id_mapping = {
    "kidney": 0,
    "bipolar_forceps": 1,
    "prograsp_forceps": 2,
    "large_needle_driver": 3,
    "monopolar_curved_scissors": 4,
    "ultrasound_probe": 5,
    "suction": 6,
    "clip_applier": 7,
    "stapler": 8,
}


random.seed(0)

class_id_to_name_mapping = dict(
    zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys())
)


def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    font_path = "InputSans-Regular.ttf"

    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:, [1, 3]] = annotations[:, [1, 3]] * w
    transformed_annotations[:, [2, 4]] = annotations[:, [2, 4]] * h

    transformed_annotations[:, 1] = transformed_annotations[:, 1] - (
        transformed_annotations[:, 3] / 2
    )
    transformed_annotations[:, 2] = transformed_annotations[:, 2] - (
        transformed_annotations[:, 4] / 2
    )
    transformed_annotations[:, 3] = (
        transformed_annotations[:, 1] + transformed_annotations[:, 3]
    )
    transformed_annotations[:, 4] = (
        transformed_annotations[:, 2] + transformed_annotations[:, 4]
    )

    font = ImageFont.truetype(font_path, 48)

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0, y0), (x1, y1)), width=5)
        plotted_image.text(
            (x0, y0 - 20), class_id_to_name_mapping[(int(obj_cls))], font=font
        )

    plt.imshow(np.array(image))
    plt.show()


# Get any random annotation file
annotation_file = random.choice(annotations)
with open(annotation_file, "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x] for x in annotation_list]

# Get the corresponding image file
image_name = os.path.splitext(annotation_file.split("/")[-1])[0] + ".png"
image_file = root_dir + annotation_file.split("/")[2] + "/left_frames/" + image_name

assert os.path.exists(image_file)

# Load the image
image = Image.open(image_file)

# Plot the Bounding Box
plot_bounding_box(image, annotation_list)
