"""
This script is used to convert mensural sources (jpg, png, pdf or csv) to MEI 
format. It also provides options to select specific pages in a pdf file for 
conversion, create a humdrum output file, and enable debug mode.

Parameters
----------
source : str
    The mensural source file (jpg, png, pdf or csv) to be converted.
pages : str
    The pages in the pdf file to be converted (e.g., '1-3,5').
humdrum : bool
    If True, a humdrum output file will be created.
debug : bool
    If True, the script will run in debug mode.

Returns
-------
None
"""
import argparse

from mensural_to_mei.configs import config
from mensural_to_mei.run_conversion import conversion_pipeline



if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='Analyzing command line arguments')

    # Add the source argument with choices for file types
    parser.add_argument('-s', '--source', type=str, dest='source', help='mensural sources (jpg, png, pdf or csv)')

    # Add the pages argument
    parser.add_argument('-p', '--pages', type=str, dest='pages', help='pages to convert in pdf-file (e.g. 1-3,5)')

    # Add humdrum argument
    parser.add_argument('-hm', '--humdrum', dest='humdrum', action='store_true', help='create humdrum output file')

    parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='debug mode')

    # Parse the arguments
    args = parser.parse_args()

    config.DEBUG_MODE = args.debug
    conversion_pipeline(
        source=args.source,
        pages=args.pages,
        humdrum=args.humdrum,
    )