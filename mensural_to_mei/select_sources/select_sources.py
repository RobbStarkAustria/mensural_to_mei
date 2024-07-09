"""
This script provides functionality to select source files for conversion
from various formats and handles the conversion of PDF files to images.
It utilizes external modules for file handling and color-coded console
output.

Functions:
    select_sources(sources: str, pages: str) -> list:
        Selects and sorts the source files for conversion based on the
        specified sources and pages. It supports single file paths,
        directory paths, or CSV file paths for sources and handles page
        selection for PDF files.

Example:
    select_sources('source_directory/', '1-3,5')

Note:
    The script requires the 'colorama' and 'termcolor' libraries for
    console output formatting. It also depends on the 'mensural_to_mei'
    package for PDF conversion and file checking utilities. The
    configuration file should be in YAML format and located in the
    'configs' directory.

Raises:
    FileNotFoundError: If the specified source files do not exist.
    SystemExit: If pages are not specified when the source is a PDF file.
"""

import csv
import glob
import os
import sys
from mensural_to_mei.configs import config
from mensural_to_mei.select_sources.convert_pdf_to_images import convert_pdf_to_images
from mensural_to_mei.utils import check_files_exist
from colorama import just_fix_windows_console
from termcolor import cprint

just_fix_windows_console()


def select_sources(sources: str, pages: str) -> list:
    """
    Selects source files for conversion based on provided sources and pages.

    Parameters
    ----------
    sources : str
        The path to the source files. This can be a single file path, a
        directory path, or a CSV file path.
    pages : str
        The pages to be converted in the PDF file. This is only applicable
        if the sources are a PDF file.

    Returns
    -------
    list
        A list of file names to be converted, sorted in alphabetical order.

    Raises
    ------
    FileNotFoundError
        If the sources file does not exist.
    SystemExit
        If pages is empty when the source is a PDF file.
    """


    # check if source files exist
    check_files_exist(sources)
    extension = os.path.splitext(sources)[1]
    
    # check if source is a pdf file
    if extension == '.pdf':
        # check if pages are specified
        if pages == '':
            cprint("Please enter pages to convert (e.g. 1-3,5) and try again.", "red")
            sys.exit(-1)

        # convert pdf to images
        convert_pdf_to_images(sources, True, pages, config.OUTPUT_FOLDERS['pdf_images'], config.OUTPUT_FOLDERS['pdf_temp'])

        # get sorted list of pdf images
        filenames = sorted(glob.glob(os.path.join(config.OUTPUT_FOLDERS['pdf_images'], "*.jpg")))
    elif extension == '.csv':
        # read sources from csv
        filenames = []
        with open(sources, 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            for row in csv_reader:
                check_files_exist(row[0])
                filenames.append(row[0])
    else:
        if extension == '.jpg' or extension == '.png':
            filenames = [sources]
        else:
            cprint("Please enter a valid source (jpg, png, pdf or csv) and try again.", "red")
            sys.exit(-1)

    if not filenames:
        cprint("No valid source files found. Please check the source and try again.", "red")
        sys.exit(-1)
        
    return filenames