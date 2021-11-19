#!/usr/bin/env python3

"""drawlinesdemo.py

The output from a human pose estimator contains a list of coordinates.
We can use python to mark these coordinates on existing images;
as well as to generate new, smoother imagery representing the human pose.

Two intuitive libraries for that are:

    - OpenCV (uses [np-]arrays to represent image data)
        docs: https://docs.opencv.org/master/

    - PIL (has its own image class)
        docs: https://pillow.readthedocs.io/en/stable/

Both libraries also allow for other image manipulation which might be useful
(text and other geometric shapes).

Note: This contains unclean code :)

Finn M Glas, 2021-09-18 16:33:00 CEST
"""

# reuseable converter functions
import numpy as np


def convertPIL2OCV(pil_image):
    """Convert PIL Image to an OpenCV Image"""

    return np.asarray(pil_image)


def convertOCV2PIL(ocv_image):
    """Convert OpenCV Image to a PIL Image"""

    if type(ocv_image) == list:
        ocv_image = np.array(ocv_image)

    return PIL.Image.fromarray(ocv_image)


# main demo

import PIL.Image
import PIL.ImageDraw

if __name__ == "__main__":

    # Variables for this demo
    width: int = 1280
    height: int = 800
    colorspace: str = "RGB"
    color: str = "white"

    linecolor: str = "red"
    linewidth: int = 2

    # Create a dummy image (to be replaced by a camera picture)
    dummy = PIL.Image.new(colorspace, (width, height), color)

    dummy = convertOCV2PIL(convertPIL2OCV(dummy))  # test conversion

    draw = PIL.ImageDraw.Draw(dummy)
    draw.line(xy=(0, 0, 1280, 800), fill=linecolor, width=linewidth)

    ## OpenCV alternative (needs image-array format and rgb-tuple colors):
    # import cv2
    # cv2.line(img, (0, 0), (1280, 800), (255, 0, 0), linewidth)

    dummy.show()  # Could also be streamed somewhere or saved
