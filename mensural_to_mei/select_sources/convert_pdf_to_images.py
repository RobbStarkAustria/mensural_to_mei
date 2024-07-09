"""
This script contains functions to convert a PDF file to a series of images.

The `convert_pdf` function takes a PDF file name, resolution in dots per inch (dpi),
output path, and pages to convert as input. It converts the specified pages of the
PDF file to images at the specified resolution and saves them to the specified
output path.

The `rename_pdf_image` function takes a PDF file name, an image file name, and a side
indicator as input. It renames the image file based on the PDF file name, the index
of the image in the PDF file, and the side indicator.

The `convert_pdf_to_images` function takes a PDF file name, a boolean indicating
whether the PDF file contains two-sided pages, pages to convert, a PDF output
folder, and a PDF temp folder as input. It converts the specified pages of the
PDF file to images and saves them to the specified output folder. The function
uses the `convert_pdf` and `rename_pdf_image` functions to perform the conversion
and renaming.

Please refer to the docstrings of the `convert_pdf`, `rename_pdf_image`, and
`convert_pdf_to_images` functions for more details about their parameters and
return values.
"""

import glob
import os
import shutil

from tqdm import tqdm
from mensural_to_mei.utils import remove_files
from pdf2image import convert_from_path


def convert_pdf(
        filename: str = "",
        dpi: int = 300,
        output_path: str = "pdf_temp",
        pages_to_convert: str = ""
) -> None:
    """
    Converts specified pages of a PDF file into JPEG images.

    This function takes a PDF file and converts specified pages into JPEG
    images at a specified resolution. The images are saved to a specified
    output path.

    Parameters
    ----------
    filename : str, optional
        The name of the PDF file to convert. Defaults to an empty string.
    dpi : int, optional
        The resolution in dots per inch at which to convert the PDF
        pages. Defaults to 300.
    output_path : str, optional
        The path where the converted images will be saved. Defaults to
        "pdf_temp".
    pages_to_convert : str, optional
        A string specifying the pages to convert, formatted as a
        comma-separated list of page ranges. For example, "1-3,5" would
        convert pages 1, 2, 3, and 5. Defaults to an empty string, which
        means all pages are converted.

    Returns
    -------
    None
    """

    remove_files(output_path)

    pdf_file = filename
    test_images = {}

    if pages_to_convert != "":
        list_of_pages = pages_to_convert.split(",")

        for i in list_of_pages:
            test_image = None
            if "-" in i:
                page_range = i.split("-")
                first_page = int(page_range[0])
                last_page = int(page_range[1])
            else:
                first_page = int(i)
                last_page = int(i)

            test_image = convert_from_path(
                pdf_file,
                first_page=first_page,
                last_page=last_page,
                fmt="jpeg",
                dpi=dpi,
                jpegopt={"quality": 95},
                output_folder=output_path,
                output_file="pdf",
                thread_count=4
            )

        return None

    test_images = convert_from_path(
        pdf_file,
        fmt="jpeg",
        dpi=dpi,
        jpegopt={"quality": 95},
        output_folder=output_path,
        output_file="pdf",
        thread_count=4
    )
    return

def rename_pdf_image(pdf_name: str, f_name: str, side_indicator="") -> str:
    """ renames pdf image """
    f_name_parts = f_name.split("-")
    index = int(f_name_parts[1][:-4])
    pdf_basename = os.path.basename(pdf_name)
    new_f_name = f"{pdf_basename[:-4]}_{index:04d}{side_indicator}.jpg"
    return new_f_name

def convert_pdf_to_images(
        pdf_name: str,
        two_sided: bool,
        pages_to_convert: str,
        pdf_output_folder: str,
        pdf_temp_folder: str) -> None:
    """
    Converts specified pages of a PDF file into images and saves them 
    to a specified folder.

    This function takes a PDF file name, a boolean indicating whether 
    the PDF file contains two-sided pages, pages to convert, a PDF 
    output folder, and a PDF temp folder as input. It converts the 
    specified pages of the PDF file to images and saves them to the 
    specified output folder. The function uses the `convert_pdf` and 
    `rename_pdf_image` functions to perform the conversion and renaming.

    Parameters
    ----------
    pdf_name : str
        The name of the PDF file to convert.
    two_sided : bool
        Whether the PDF file contains two-sided pages.
    pages_to_convert : str
        A string specifying the pages to convert, formatted as a 
        comma-separated list of page ranges.
    pdf_output_folder : str
        The folder where the converted images will be saved.
    pdf_temp_folder : str
        The folder where temporary files will be saved during the 
        conversion process.

    Returns
    -------
    None
    """
    
    remove_files(pdf_output_folder)

    pdf_pages = convert_pdf(
        pdf_name, pages_to_convert=pages_to_convert, output_path=pdf_temp_folder)
    
    pdf_pages = sorted(glob.glob(os.path.join(pdf_temp_folder, "*.jpg")))

    for f_name in tqdm(pdf_pages, "move to pdf-folder"):
        new_f_name = rename_pdf_image(pdf_name, f_name)
        shutil.move(f_name, os.path.join(pdf_output_folder, new_f_name))
    
    return
