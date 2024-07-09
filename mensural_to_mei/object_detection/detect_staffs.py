"""
Staff Detection in Musical Notation Images

This script provides functionality to detect staff lines in images of musical
notation using a pre-trained ONNX model. It includes the necessary imports,
preprocessing steps, inference, and post-processing to accurately identify
and return the coordinates of staff lines within an image.

The main function, `detect_staffs`, takes an image as input and returns a
sorted list of detected staff lines. The script also includes utility functions
to preprocess the image and analyze the output from the ONNX model.

Functions:
    detect_staffs(image): Detects and returns the sorted staff lines in the
        given musical notation image.

Utility Modules:
    mensural_to_mei.object_detection.do_inference: Performs inference on the
        preprocessed image.
    mensural_to_mei.preprocess_images.preprocess_images: Preprocesses the
        input image for inference.
    mensural_to_mei.utils: Contains the function to analyze the output from
        the ONNX model.

Example:
    To use the `detect_staffs` function to detect staff lines in an image:

        from detect_staffs_script import detect_staffs
        image = ...  # Load your image as a numpy array
        detected_staffs = detect_staffs(image)
        print(detected_staffs)

Note:
    Ensure that the ONNX model path is correctly set and that all dependencies
    are installed before running the script.
"""

import math
import numpy as np
import onnxruntime as ort
from mensural_to_mei.configs import config
from mensural_to_mei.object_detection.do_inference import do_inference
from mensural_to_mei.preprocess_images.preprocess_images import process_image
from mensural_to_mei.utils import do_onnx_analysis

def detect_staffs(image: np.ndarray) -> list[tuple]:
    """
    Detects staff lines in a musical notation image using a pre-trained ONNX model.

    Parameters
    ----------
    image : np.ndarray
        The input image as a numpy array.

    Returns
    -------
    list of tuple
        A sorted list of detected staff lines. Each staff line is represented
        as a tuple of four integers: (x_min, y_min, x_max, y_max).

    Raises
    ------
    None

    Examples
    --------
    >>> image = ...  # Load your image as a numpy array
    >>> detected_staffs = detect_staffs(image)
    >>> print(detected_staffs)
    [(x_min1, y_min1, x_max1, y_max1), (x_min2, y_min2, x_max2, y_max2), ...]
    """


    onnx_model_path = config.MODEL_PATHES['staffs']

    fullpage_image, padding, resize_factor = process_image(image, (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
                                                           rescale=True)
    
    session = ort.InferenceSession(onnx_model_path,
                                #    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                                   )

    img = fullpage_image.astype(np.float32)

    output = do_inference(session, img)
    
    staffs, _ = do_onnx_analysis(output)    # labels not needed

    offset_y = padding[0]
    offset_x = padding[2]

    security_range = 20

    # recalculate detected coordinates to original dimensions
    for staff in staffs:
        staff[0] = max(0, math.floor((staff[0] - offset_x) / resize_factor))
        staff[1] = max(0, math.floor((staff[1] - offset_y - security_range) / resize_factor))
        staff[2] = min(image.shape[1], math.floor((staff[2] - offset_x) / resize_factor))
        staff[3] = min(image.shape[0], math.floor((staff[3] - offset_y + security_range) / resize_factor))

    # sort staffs in ascending order by ymin-coordinate to get staffs in correct order
    sorted_staffs = sorted(staffs, key=lambda box: box[1])

    return sorted_staffs