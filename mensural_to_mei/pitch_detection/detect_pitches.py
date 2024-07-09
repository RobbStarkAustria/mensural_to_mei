
"""
This script contains a function to detect pitches in musical symbols 
using ONNX models.

The `detect_pitches` function takes a list of detected symbols as input 
and uses three different ONNX models to detect pitches in the symbols. 
The function returns a list of detected pitch for each symbol.

The `detect_pitches` function uses the following ONNX models for pitch 
detection:
- "models/classification/best_clef.onnx"
- "models/classification/best_all_symbols.onnx"
- "models/classification/best_mensuration.onnx"

The function also uses the following yaml files for classes:
- "models/classification/best_clef.yaml"
- "models/classification/best_all_symbols.yaml"
- "models/classification/best_mensuration.yaml"

Please refer to the docstring of the `detect_pitches` function for more 
details about its parameters and return value.
"""
import itertools
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from mensural_to_mei.configs import config
from mensural_to_mei.object_detection.do_inference import do_inference
from mensural_to_mei.preprocess_images.preprocess_images import calc_new_dimensions
from mensural_to_mei.utils import load_program_folders, load_yaml
from colorama import just_fix_windows_console
from termcolor import cprint

just_fix_windows_console()

def detect_pitches(detected_symbols: list) -> list:
    """
    Detects pitches in the given symbols using ONNX models.

    This function uses three different ONNX models to detect pitches in 
    the symbols. The function returns a list of detected pitches for each 
    symbol.

    Parameters
    ----------
    detected_symbols : list
        List of detected symbols. Each symbol is represented as a 
        dictionary with 'type' and 'pitch' keys.

    Returns
    -------
    list
        A list of dictionaries. Each dictionary contains the 'type' and 
        'pitch' of a symbol.

    Note
    ----
    Pathes to models and classes are defined in config.py and should be
    defined properly.
    """
    CLASSES_CLEF = load_yaml(config.LABEL_PATHES['clef'])
    CLASSES_ALL_SYMBOLS = load_yaml(config.LABEL_PATHES['all_symbols'])
    CLASSES_MENS = load_yaml(config.LABEL_PATHES['mensuration'])

    session_clef = ort.InferenceSession(config.MODEL_PATHES['clef'])
    session_all_symbols = ort.InferenceSession(config.MODEL_PATHES['all_symbols'])
    session_mens = ort.InferenceSession(config.MODEL_PATHES['mensuration'])

    PITCH_DETECT_LIST = ['ma-u', 'ma-d', 'lo-u', 'lo-d', 'bre', 'sebre',
                         'mi-u', 'mi-d', 'sm-u', 'sm-d', 'fu-u', 'fu-d',
                         "sf-u", 'sf-d', 'br-min', 'sb-min', "li-lolu"]
    CLASSIFICATION_LIST = PITCH_DETECT_LIST + ['clef', 'mens']

    symbol_pitch_list = {}
    for sourcefile, symbols in detected_symbols.items():
        cprint(f"Classify pitches in {sourcefile}", "blue")
        start = time.time()
        flattened_list = list(itertools.chain.from_iterable(symbols))

        preprocessed_path = os.path.join(config.OUTPUT_FOLDERS["preprocessed_images"], f"{sourcefile}.jpg")
        grayscale_image = cv2.imread(preprocessed_path)

        staff_symbol_list = []
        classified_symbols = 0
        for staff_symbols in tqdm(symbols, desc='detect pitches'):
            symbol_list = []
            for roi in staff_symbols:
                pitch =''
                note_type = roi[4]

                # define short_type for symbols not in classification_list
                # for classified notes the short_type will be changed after classification
                short_type = note_type

                if note_type in CLASSIFICATION_LIST:
                    classified_symbols += 1
                    # Extract the region of interest from the image
                    roi_image = grayscale_image[roi[1]:roi[3], roi[0]:roi[2]]
                    new_width, new_height, padding, resize_factor = calc_new_dimensions(
                        roi_image, (config.SYMBOL_SIZE[0], config.SYMBOL_SIZE[1])
                    )
                    # Resize the region of interest (roi) image to the new dimensions using Lanczos resampling.
                    # Lanczos resampling is a high-quality resampling method for digital images.
                    resized_roi = cv2.resize(roi_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

                    # Create a new image of shape (224, 224, 3) filled with zeros. The dtype is set to np.uint8 
                    # because it's the standard for images, where each pixel is represented by an 8-bit (one byte) integer.
                    processed_image = np.zeros((config.SYMBOL_SIZE[0], config.SYMBOL_SIZE[1], 3), dtype=np.uint8)

                    # Fill the processed_image array with 255. In terms of images, 255 represents white when using 
                    # an 8-bit grayscale image. So, this line effectively makes the entire image white.
                    processed_image.fill(255)

                    # Copy the resized region of interest to the center of the new image.
                    # The padding values are used to determine the location of the center of the image.
                    processed_image[padding[0]:padding[1], padding[2]:padding[3]] = resized_roi


                    if note_type in PITCH_DETECT_LIST:
                        output = do_inference(session_all_symbols, processed_image)
                        pitch = CLASSES_ALL_SYMBOLS[np.argmax(output[0])]
                    elif note_type=='clef':
                        output = do_inference(session_clef, processed_image)
                        pitch = CLASSES_CLEF[np.argmax(output[0])]
                    elif note_type=='mens':
                        output = do_inference(session_mens, processed_image)
                        pitch = CLASSES_MENS[np.argmax(output[0])]
                    
                    short_type = str.split(note_type, "-")[0]

                symbol = {
                    'type': short_type,
                    'pitch': pitch
                }
                symbol_list.append(symbol)

            staff_symbol_list.append(symbol_list)

        symbol_pitch_list[sourcefile] = staff_symbol_list
        os.remove(preprocessed_path)

        end = time.time()
        cprint(f"Classified {classified_symbols} pitches in {end - start} seconds", "green")

    return symbol_pitch_list
