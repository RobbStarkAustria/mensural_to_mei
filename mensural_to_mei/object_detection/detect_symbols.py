"""
This script contains a function to detect symbols in an image using an 
ONNX model. The function, `detect_symbols`, takes a list of staffs, an 
image, and a list of classes as input, and returns a list of detected 
symbols for each staff.

The script imports necessary modules and functions from `math`, `os`, 
`cv2`, `numpy`, `onnxruntime`, and other custom modules.

The `detect_symbols` function uses the ONNX model located at 
"models/object_detection/best_symbols.onnx" for inference. It processes 
each staff in the input list, resizes the staff image, performs 
inference, and analyzes the output to detect symbols. The detected 
symbols are then sorted and appended to the `staff_symbols` list which 
is returned by the function.

Please refer to the docstring of the `detect_symbols` function for more 
details about its parameters and return value.
"""

import math
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from mensural_to_mei.configs import config
from mensural_to_mei.object_detection.do_inference import do_inference
from mensural_to_mei.preprocess_images.preprocess_images import calc_new_dimensions
from mensural_to_mei.utils import do_onnx_analysis


def detect_symbols(
        staffs: list,
        image: np.ndarray,
        classes: list
) -> list:
    """
    Detects symbols in the given image.

    This function uses an ONNX model to detect symbols in the image. The 
    symbols are detected for each staff in the staffs list. The function 
    returns a list of detected symbols for each staff.

    Parameters
    ----------
    staffs : list
        List of staffs in the image. Each staff is represented as a list 
        of four integers [x1, y1, x2, y2] representing the coordinates of 
        the bounding box of the staff in the image.
    image : np.ndarray
        The image in which to detect symbols. The image is a numpy array 
        of shape (H, W, 3), where H is the height of the image, W is the 
        width of the image, and 3 represents the three color channels.
    classes : list
        List of classes for the symbols to be detected.

    Returns
    -------
    list
        A list of lists. Each inner list contains the detected symbols for 
        a staff. Each symbol is represented as a list of five elements 
        [x1, y1, x2, y2, class], where [x1, y1, x2, y2] represents the 
        coordinates of the bounding box of the symbol in the image and 
        class is the class of the symbol.
    """
    onnx_model_path = config.MODEL_PATHES['symbols']

    session = ort.InferenceSession(onnx_model_path,
                                #    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                                   )

    staff_symbols = []
    for staff in tqdm(staffs, desc='Detecting symbols'):
        staff_image = image[staff[1]:staff[3], staff[0]:staff[2]]

        new_w, new_h, padding, resize_factor = calc_new_dimensions(staff_image, (config.STAFF_SIZE[0], config.STAFF_SIZE[1]))

        resized_image = cv2.resize(staff_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        staff_image = np.zeros((config.STAFF_SIZE[1], config.STAFF_SIZE[0], 3), dtype=np.uint8)
        staff_image.fill(255)
        staff_image[padding[0]:padding[1], padding[2]:padding[3]] = resized_image
        
        output = do_inference(session, staff_image)

        boxes, labels = do_onnx_analysis(output)

        symbol_list = [[box[0], box[1], box[2], box[3], classes[label]] for box, label in zip(boxes, labels) if label in classes]

        for symbol in symbol_list:
            symbol[0] = max(0, math.floor((symbol[0] - padding[2]) / resize_factor + staff[0]))
            symbol[1] = max(0, math.floor(staff[1]))
            symbol[2] = min(image.shape[1], math.floor((symbol[2] - padding[2]) / resize_factor + staff[0]))
            symbol[3] = min(image.shape[0], math.floor(staff[3]))

        sorted_symbols = sorted(symbol_list, key=lambda box: box[0])

        staff_symbols.append(sorted_symbols)
    

    return staff_symbols

