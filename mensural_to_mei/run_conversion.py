"""
Conversion Pipeline Script.

This script defines a conversion pipeline that processes images of
mensural notation and converts them to MEI (Music Encoding Initiative)
and Humdrum formats. It includes functions for selecting sources,
detecting symbols, detecting pitches, and the conversion process.

Functions
---------
- check_for_output_folders()
    Checks for the existence of output folders and creates them if 
    they do not exist.
- select_sources(source, pages)
    Selects image sources based on provided directory and page numbers.
- do_detection(image_sources)
    Detects musical symbols in the given image sources.
- detect_pitches(all_found_symbols)
    Detects pitches from the detected musical symbols.
- conversion_pipeline(source='', pages='', humdrum=False, debug=False)
    Main function that orchestrates the conversion pipeline.

Script Usage
------------
The script is executed with a main function call to `conversion_pipeline`
with the desired parameters. If `debug` is set to True, the detected
symbols and pitches are saved to a pickle file for debugging purposes.

Example
-------
To run the conversion pipeline with debugging enabled, saving the output
to 'symbols_and_pitches.pkl':

    conversion_pipeline(source='path/to/images', debug=True)

Notes
-----
The script requires the 'pickle' module for debugging purposes and
functions from the 'mensural_to_mei' package for the conversion process.
"""

import os
import pickle
import time

from colorama import just_fix_windows_console
from termcolor import cprint
from mensural_to_mei.configs import config
from mensural_to_mei.convert_detections.convert_to_mei_and_humdrum import convert_to_mei_and_humdrum
from mensural_to_mei.object_detection.do_detection import do_detection
from mensural_to_mei.pitch_detection.detect_pitches import detect_pitches
from mensural_to_mei.select_sources.select_sources import select_sources

just_fix_windows_console()

def check_for_output_folders() -> None:
    """
    Checks for the existence of output folders and creates them if 
    they do not exist.

    This function iterates over the output folders specified in the 
    global `config.OUTPUT_FOLDERS` dictionary. If an output folder 
    does not exist, the function creates it and prints a message.

    Returns
    -------
    None
    """
    for output_folder in config.OUTPUT_FOLDERS.values():
        if not os.path.exists(output_folder):
            cprint(f"Creating folder: {output_folder}", "green", "on_white")
            os.makedirs(output_folder)

def conversion_pipeline(
        source: str = '',
        pages: str = '',
        humdrum: bool= False) -> dict:
    
    check_for_output_folders()

    start_time = time.time()
    
    image_sources = select_sources(source, pages)
    
    all_found_symbols = do_detection(image_sources)

    symbols_and_pitches = detect_pitches(all_found_symbols)
    
    if config.DEBUG_MODE:
        # save detection for debug of following function to avoid unnecessary detection steps
        with open("symbols_and_pitches.pkl", "wb") as f:
            pickle.dump(symbols_and_pitches, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    convert_to_mei_and_humdrum(symbols_and_pitches, humdrum)

    end_time = time.time()
    cprint(f"Conversion took {end_time - start_time:.2f} seconds", "black", "on_white")

    return symbols_and_pitches