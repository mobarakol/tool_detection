from os.path import dirname, realpath
import xml.etree.cElementTree as et
import os
from os.path import join, exists, isfile
import json
from typing import Tuple
from glob import glob

# Paths
ROOT_DIR = dirname(realpath(__file__))
data_dir = join(ROOT_DIR, "../dataset")
train_dir = join(data_dir, "train")
val_dir = join(data_dir, "val")

# Files to create, just path definition
filename_json = "dataset.json"
file_train = join(train_dir, filename_json)
file_val = join(val_dir, filename_json)

def get_points(x, x2, y, y2) -> dict:
    """
    To create rect shape, all points included in dictionary.

    :params **args: int values, to create polygon shape
    :return: A dictionary
    """
    regions = (
        {"x": x,
         "y": y,
         "x2": x2,
         "y2": y2
         })
    return regions


def calculate_xy(x_max: int, x_min: int, y_max: int, y_min: int) -> Tuple[int, int]:
    """
    Formula to get X and Y values.

    X = x_min + (x_max-x_min)/2
    Y = y_min + (y_max-y_min)/2

    :return: A tuple, X and Y values, int values
    """
    x_min_tmp = int(x_max - x_min) / 2
    x_value = int(x_min + x_min_tmp)

    y_min_temp = int(y_max - y_min) / 2
    y_value = int(y_min + y_min_temp)

    return x_value, y_value


def convert_xml_to_json(path: str, image_list: list, xml_list: list, mode):
    """
    Convert from xml to json. For each img is necessary
    a xml file annotation. This function save json file.
    :param path: A str like /to/path/file
    :param image_list: A list of images in the same path
    """
    all_json = {}
    xml_dir = xml_list + "/" + mode
    path = xml_dir

    for img in image_list:
        name_xml = os.path.splitext(img.split("/")[-1])[0] + '.xml'
        images = ({"filename": img})
        try:
            root = et.ElementTree(file=join(path, name_xml)).getroot()
        except:
            print(join(path, name_xml), " FAILED")  
            continue  
        obj_counter, regi = {}, {}
        number = 0
        for child_of_root in root:
            if child_of_root.tag == 'objects':
                for child_of_object in child_of_root:
                    if child_of_object.tag == 'name':
                        obj_id = child_of_object.text.split(' ')[0] 
                        obj_counter[obj_id] = number
                    if child_of_object.tag == 'bndbox':
                        for child_of_root in child_of_object:
                            # print(child_of_root.tag, child_of_root.text)
                            if child_of_root.tag == 'xmin':
                                x_min = int(child_of_root.text)
                            if child_of_root.tag == 'xmax':
                                x_max = int(child_of_root.text)
                            if child_of_root.tag == 'ymin':
                                y_min = int(child_of_root.text)
                            if child_of_root.tag == 'ymax':
                                y_max = int(child_of_root.text)

                # x_value, y_value = calculate_xy(x_max, x_min, y_max, y_min)
                coord = get_points(x_min, x_max, y_min, y_max)
                # print(coord)

                regions = ({"region_attributes": {"name": obj_id}})
                regions.update({"shape_attributes": coord})
                regions.update({"name": "rect"})
                regi[number] = regions.copy()
                regions = {"regions": regi}
                images.update(regions)
                images.update({"size": os.path.getsize(img)})
                all_json[img] = images.copy()
                number += 1

    out_file = open(join('dataset/processed/json', mode + "_" + filename_json), "a")
    json.dump(all_json, out_file)
    print("File dataset.json was save in: ", join('dataset/processed/json', mode + "_" + filename_json))

if __name__ == "__main__":
    
    images_train = glob("dataset/processed/images/train/*.png")
    images_val = glob("dataset/processed/images/val/*.png")

    xml_train = glob("dataset/processed/labels_xml/train/*.xml")
    xml_val = glob("dataset/processed/labels_xml/val/*.xml")
    xml_dir = "dataset/processed/labels_xml/"

    # Convert from xml to json in both train and val
    convert_xml_to_json(train_dir, images_train, xml_dir, "train")
    convert_xml_to_json(val_dir, images_val, xml_dir, "val")