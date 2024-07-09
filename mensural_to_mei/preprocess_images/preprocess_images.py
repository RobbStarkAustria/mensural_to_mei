"""
This script provides a suite of functions for image preprocessing, including
noise removal, resizing, and normalization. It is designed to work with
images in a numpy array format and utilizes OpenCV for image operations.

The script includes the following functions:
- remove_noise: Applies Fast Non-Local Means Denoising to remove noise from images.
- calc_new_dimensions: Calculates new dimensions for an image to fit a specified size while maintaining aspect ratio.
- process_image: Processes an image by converting it to grayscale, denoising, normalizing, and resizing.
- preprocess_images: Preprocesses a list of images according to configurations specified in a YAML file.

Requirements:
- OpenCV
- NumPy
- YAML configuration file for image preprocessing parameters

Example:
    >>> image_sources = ['image1.jpg', 'image2.png']
    >>> preprocess_images(image_sources)
    True

Note:
    The script assumes that the configuration file 'config_images.yaml' is located in the 'configs' directory.
"""

import cv2
import numpy as np
from mensural_to_mei.utils import load_configs
import os

def remove_noise(image: np.ndarray) -> np.ndarray:
    """
    Applies Fast Non-Local Means Denoising to an image to remove noise.

    Parameters
    ----------
    image : np.ndarray
        The input image to be denoised.

    Returns
    -------
    np.ndarray
        The denoised image.

    Notes
    -----
    The function uses the Fast Non-Local Means Denoising algorithm, which
    works by considering a template patch around each pixel and computing
    a weighted average of similar patches found in a larger search window.

    The parameters for the algorithm are:
    - h: The filtering strength. Higher values remove noise but also remove
    image details. If None, the value is calculated as 3*sigma + 1.
    - templateWindowSize: Size of the template patch used for denoising.
    Should be odd and greater than h.
    - searchWindowSize: Size of the window used to search for patches.
    Should be odd and greater than templateWindowSize.
    - normType: Type of norm used for denoising. Options are cv2.NORM_L1,
    cv2.NORM_L2, or cv2.NORM_MINMAX.

    Example
    -------
    >>> image = cv2.imread('path/to/noisy_image.jpg')
    >>> denoised_image = remove_noise(image)
    >>> cv2.imwrite('path/to/denoised_image.jpg', denoised_image)
    """

    return cv2.fastNlMeansDenoising(image, None, 10, 10, cv2.NORM_MINMAX)

def calc_new_dimensions(img: np.ndarray, new_size: tuple) -> tuple:
    """
    Calculates the new dimensions of an image after resizing it to fit within
    a specified size while maintaining its aspect ratio.

    Parameters
    ----------
    img : np.ndarray
        The input image as a numpy array.
    new_size : tuple
        The desired size (width, height) for the output image.

    Returns
    -------
    tuple
        A tuple containing:
        - new_width (int): The new width of the resized image.
        - new_height (int): The new height of the resized image.
        - padding (tuple): A tuple representing the number of pixels to add
        to the top, bottom, left, and right of the image to center it
        within the new size.
        - resize_factor (float): The smallest scaling factor applied to the
        image to fit it within the new size.

    Notes
    -----
    The function calculates the resize factor based on the desired size and
    the original image dimensions. It then computes the new dimensions,
    padding, and resize factor. If the image is not rescaled, the padding
    and resize factor are set accordingly.

    Example
    -------
    >>> img = np.random.rand(480, 640)  # Example input image
    >>> new_size = (800, 600)
    >>> new_width, new_height, padding, resize_factor = calc_new_dimensions(img, new_size)
    >>> print(new_width, new_height, padding, resize_factor)
    800 600 (0, 600, 80, 720) 1.25
    """

    h, w = img.shape[:2]

    resize_w = new_size[0] / w
    resize_h = new_size[1] / h

    resize_factor = min(resize_h, resize_w)

    new_width = np.ceil(min(w * resize_factor, new_size[0])).astype(int)
    new_height = np.ceil(min(h * resize_factor, new_size[1])).astype(int)

    offset_w = (new_size[0] - new_width) // 2
    offset_h = (new_size[1] - new_height) // 2
    padding = (offset_h, offset_h + new_height, offset_w, offset_w + new_width)

    return new_width, new_height, padding, resize_factor

def process_image(
        source_img: np.ndarray,
        new_size: tuple,
        rescale: bool=True,
) -> tuple:
    """
    Processes an image by converting it to grayscale, denoising, normalizing,
    and resizing based on the given parameters.

    Parameters
    ----------
    source_img : np.ndarray
        The source image to be processed.
    new_size : tuple
        The desired size (width, height) for the output image.
    rescale : bool, optional
        A flag that determines whether to resize the image or not. The default
        is True.

    Returns
    -------
    tuple
        A tuple containing the processed image as a numpy array, the padding
        applied to the image, and the resize factor used. The padding is a
        tuple of four integers (top, bottom, left, right), and the resize
        factor is a float indicating how much the image was scaled.

    Notes
    -----
    The function first converts the source image to grayscale and applies
    denoising. It then normalizes the pixel values. If rescaling is enabled,
    it calculates the new dimensions to maintain the aspect ratio and resizes
    the image accordingly. The resized image is centered and padded to match
    the new size.

    Example
    -------
    >>> source_img = cv2.imread('path/to/image.jpg')
    >>> new_size = (800, 600)
    >>> processed_img, padding, resize_factor = process_image(source_img, new_size)
    >>> print(processed_img.shape)
    (600, 800, 3)
    """

    grayscale_image = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

    denoized_image = remove_noise(grayscale_image)

    norm_img = np.zeros((source_img.shape[0], source_img.shape[1]))
    normed = cv2.normalize(denoized_image, norm_img, 0, 255, cv2.NORM_MINMAX)

    if rescale:
        new_width, new_height, padding, resize_factor = calc_new_dimensions(
            normed, new_size
        )
        resized_image = cv2.resize(normed, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        processed_image = np.zeros(new_size, dtype=np.uint8)
        processed_image.fill(255)
        # copy the resized image to the center of the new image
        processed_image[padding[0]:padding[1], padding[2]:padding[3]] = resized_image
    else:
        processed_image = normed
        padding = (0, 0, 0, 0)
        resize_factor = 1

    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    
    return processed_image, padding, resize_factor
