"""
Symbol Detection in Musical Notation Images

This script is designed to detect musical symbols in images of musical
notation. It leverages a combination of image processing and object
detection techniques to identify and classify musical symbols from
preprocessed images.

The main function, `do_detection`, accepts a list of image file paths
and returns a dictionary mapping each image file name to its detected
musical symbols. The script employs several utility functions from the
`mensural_to_mei` package for image preprocessing, staff detection, and
symbol detection.

Functions:
    do_detection(list_of_images): Processes a list of images and detects
        musical symbols, returning a dictionary of results.

Utility Modules:
    mensural_to_mei.preprocess_images.preprocess_images: Contains the
        `process_image` function to prepare images for detection.
    mensural_to_mei.object_detection.detect_staffs: Includes the
        `detect_staffs` function to find staff lines in images.
    mensural_to_mei.object_detection.detect_symbols: Provides the
        `detect_symbols` function to classify symbols on staff lines.
    mensural_to_mei.utils: Offers additional functions like
        `count_elements`, `load_program_folders`, and `load_yaml` for
        various utility purposes.

Example:
    To use the `do_detection` function to detect symbols in a list of images:

        from symbol_detection_script import do_detection
        list_of_images = [...]  # List of image file paths
        detected_symbols = do_detection(list_of_images)
        for filename, symbols in detected_symbols.items():
            print(f"{filename}: {symbols}")

Note:
    Ensure that the configuration files for the object detection models
    and program settings are correctly set up and that all dependencies are
    installed before executing the script.
"""

import os
import time
import cv2
from mensural_to_mei.configs import config
from mensural_to_mei.preprocess_images.preprocess_images import process_image
from mensural_to_mei.object_detection.detect_staffs import detect_staffs
from mensural_to_mei.object_detection.detect_symbols import detect_symbols
from mensural_to_mei.utils import count_elements, load_yaml
from colorama import just_fix_windows_console
from termcolor import cprint

just_fix_windows_console()


def do_detection(list_of_images: list) -> dict:
    """
    Detect musical symbols in a list of images.

    This method processes each image in the provided list to detect
    musical symbols using object detection models. It outputs a
    dictionary mapping each image filename to the detected symbols.

    Parameters
    ----------
    list_of_images : list
        A list of file paths to the images that will be processed.

    Returns
    -------
    dict
        A dictionary where each key is the filename of an image (without
        the file extension) and each value is the list of detected
        symbols for that image.

    Notes
    -----
    The method utilizes pre-trained models defined in 'best_symbols.yaml'
    for object detection and 'configs.yaml' for program configurations.
    It also prints the processing status and results to the console.
    """

    SYMBOL_CLASSES = load_yaml(config.LABEL_PATHES['symbols'])

    all_found_symbols = {}
    for image_path in list_of_images:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        cprint(f"Processing {filename}", "blue")
        start = time.time()

        image = cv2.imread(image_path)
        cprint('detecting staffs', 'blue')
        staffs = detect_staffs(image)

        grayscale_image, _, _ = process_image(image, (0, 0), rescale=False)

        temp_path = os.path.join(config.OUTPUT_FOLDERS['preprocessed_images'], f"{filename}.jpg")

        cv2.imwrite(temp_path, grayscale_image)

        file_symbols = detect_symbols(staffs, grayscale_image, SYMBOL_CLASSES)
        number_of_staffs = len(file_symbols)
        number_of_symbols = count_elements(file_symbols)

        end = time.time()
        cprint(f"Found {number_of_symbols} symbols in {number_of_staffs} staffs in {end - start} seconds", "green")
        all_found_symbols[filename] = file_symbols

    return all_found_symbols