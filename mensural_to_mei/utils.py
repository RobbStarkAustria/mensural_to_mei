"""
This script provides a collection of utility functions for file management,
random data generation, image processing, and XML handling. It includes
functions to load YAML configurations, check file existence, remove files,
perform non-maximum suppression on object detection results, analyze ONNX
model outputs, count elements in nested lists, generate random strings and
numbers, pretty print XML elements, and convert dictionaries to combined
lists with metadata tracking.

Functions:
    load_yaml(file_path): Loads a YAML file and returns its contents as a
        dictionary.
    check_files_exist(file): Checks if a file exists and raises a
        FileNotFoundError if not.
    remove_files(folder_path): Removes all files within a specified folder
        path.
    load_program_folders(config_path): Loads program folder paths from a
        'configs.yaml' file.
    load_configs(config_path): Loads configuration settings from a
        'configs.yaml' file.
    do_onnx_analysis(output): Analyzes ONNX model output and performs NMS on
        the results.
    count_elements(nested_list): Counts the number of elements in a nested
        list.
    generate_random_string(length): Generates a random string of a specified
        length.
    generate_random_numbers(num_elements): Generates a list of unique random
        numbers.
    prettyprint(element, **kwargs): Prints an XML element in a formatted and
        readable manner.
    convert_to_combined_list_with_metadata(d): Converts a dictionary of lists
        into a combined list with corresponding metadata.

The script is designed to be modular and reusable for various tasks that
require data manipulation and analysis, particularly in machine learning
workflows.

Example:
    # Load configurations and perform image processing on model outputs
    configs = load_configs('path/to/configs.yaml')
    output = model_inference_results
    nms_boxes, nms_labels = do_onnx_analysis(output)
    print(f"Processed {len(nms_boxes)} bounding boxes with labels: {nms_labels}")

Note:
    The script assumes the presence of certain libraries such as 'numpy',
    'yaml', 'itertools', and 'lxml'. It also assumes
    a specific project structure for loading configurations.
"""


import glob
import os
import random
import string
import cv2
import numpy as np
import yaml
import itertools

from lxml import etree


def load_yaml(file_path: str) -> dict:
    """ load yaml form given path """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def check_files_exist(file: str) -> None:
    """ check if file exists """
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} does not exist.")


def remove_files(folder_path: str) -> None:
    """ remove all files in a folder """
    del_path = os.path.join(folder_path, "**.*")
    files = glob.glob(del_path)
    for f in files:
        os.remove(f)

def load_program_folders(config_path: str) -> dict:
    """ load program folders from configs.yaml """
    configs = load_configs(config_path)
    program_folders = configs['output_folders']
    return program_folders

def load_configs(config_path: str) -> dict:
    """ load configs from configs.yaml """
    project_root = "mensural_to_mei"
    config_path = os.path.join(project_root, "configs", "configs.yaml")

    configs = load_yaml(config_path)
    return configs

def do_onnx_analysis(output: list) -> list:
    """
    Analyzes ONNX model output and performs non-maximum suppression (NMS).

    Parameters
    ----------
    output : list
        The raw output from an ONNX model inference.

    Returns
    -------
    list
        - nms_boxes: Post-NMS bounding boxes.
        - nms_labels: Corresponding labels for the NMS boxes.

    Notes
    -----
    The function processes the model output to extract scores and class IDs,
    filters predictions based on a score threshold, and applies NMS to
    reduce overlapping boxes.

    Example
    -------
    >>> output = [onnx_model_output]
    >>> nms_boxes, nms_labels = do_onnx_analysis(output)
    >>> print(nms_boxes, nms_labels)
    [[x_min, y_min, x_max, y_max], ...] [class_id1, class_id2, ...]
    """

    outputs = np.transpose(np.squeeze(output[0]))
    scores = np.max(outputs[:, 4:], axis=1)
    # select the boxes with scores > 0.5
    predictions = outputs[scores > 0.5, :]
    scores = scores[scores > 0.5]
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    boxes = predictions[:, :4]
    conv_boxes = convert_boxes_from_cxcywh_to_xywh(boxes)

    remaining_boxes = cv2.dnn.NMSBoxes(conv_boxes, scores, 0.5, 0.5)

    nms_boxes = conv_boxes[remaining_boxes].astype(int)
    nms_labels = class_ids[remaining_boxes].tolist()

    nms_boxes = convert_boxes_from_xywh_to_xyxy(nms_boxes).tolist()

    return nms_boxes, nms_labels


def count_elements(nested_list: list) -> int:
    """ Counts the number of elements in a nested list """
    return len(list(itertools.chain.from_iterable(nested_list)))


def generate_random_string(length: int) -> str:
    """ Generates a random string of lettersof the specified length. """ 
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def generate_random_numbers(num_elements: int) -> list:
    """ Generates a list of random numbers of the specified length. """
    # Initialize an empty set to store unique random numbers
    numbers = set()
    while len(numbers) < num_elements:
        # Generate a random number by choosing 10 digits at random and joining them to form a string
        random_number = "".join(random.choices(string.digits, k=10))
        numbers.add(random_number)
    return list(numbers)


def prettyprint(element: etree.Element, **kwargs) -> None:
    """ Prints an XML element in a pretty way. """
    xml = etree.tostring(element, pretty_print=True, **kwargs)
    print(xml.decode(), end="")


def convert_to_combined_list_with_metadata(d: dict) -> tuple:
    """ Converts a dictionary of lists to a combined list and a dictionary of metadata. """
    combined_list = []
    metadata = {}
    counter = 0
    for k, v in d.items():
        for s in v:
            combined_list.append(s)
            metadata[counter] = k
            counter += 1

    return combined_list, metadata


def convert_boxes_from_cxcywh_to_xyxy(boxes):
    """ Converts boxes from cxcywh to xyxy format. """
    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    converted_boxes = np.column_stack((x_min, y_min, x_max, y_max))
    return converted_boxes

def convert_boxes_from_cxcywh_to_xywh(boxes):
    """ Converts boxes from cxcywh to xywh format. """
    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    x = x_center - width / 2
    y = y_center - height / 2

    converted_boxes = np.column_stack((x, y, width, height))
    return converted_boxes

def convert_boxes_from_xywh_to_xyxy(boxes):
    """ Converts boxes from xywh to xyxy format. """
    x = boxes[:, 0]
    y = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    xmin = x
    ymin = y
    xmax = x + width
    ymax = y + height

    converted_boxes = np.column_stack((xmin, ymin, xmax, ymax))
    return converted_boxes